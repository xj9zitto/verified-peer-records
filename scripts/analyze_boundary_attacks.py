"""
analyze_boundary_attacks.py

Summarises run_boundary_attacks.py: per-arm accuracy, detection
statistics, and cluster-bootstrap paired differences for each attack
family.

The headline quantity is the paired difference T4 - T2, the value of
content-bound verification.  Reported against the in-scope attack it
reproduces the transfer study; reported against the two out-of-scope
attacks it measures what Section IV-D excludes.  The detection columns
are the mechanism-level counterpart: an attack the predicate never fires
on cannot be defended against, whatever the accuracy column says.

Intervals are cluster bootstraps over scenarios, carrying all of a
scenario's seed-level observations together, exactly as in
analyze_transfer_multiseed.py.

Output: results/v14_boundary_attacks_analysis.json
"""

from __future__ import annotations
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

RESULTS = Path("results")
IN_FILE = RESULTS / "v14_boundary_attacks_results.json"
OUT     = RESULTS / "v14_boundary_attacks_analysis.json"

N_BOOT     = 10000
SEED       = 42
ACTIONABLE = ("replace", "urgent")
ARMS       = ["T0", "T2", "T3", "T4"]
PAIRS      = [("T4", "T2"), ("T4", "T3"), ("T3", "T2"), ("T2", "T0")]


def cluster_bootstrap(a: Dict[str, List[int]], b: Dict[str, List[int]],
                      n_boot: int = N_BOOT, seed: int = SEED
                      ) -> Dict[str, float]:
    """CI for mean(a) - mean(b), resampling scenarios (clusters)."""
    keys = sorted(set(a) & set(b))
    diffs = {k: st.mean(a[k]) - st.mean(b[k]) for k in keys}
    point = st.mean(diffs.values()) if keys else 0.0
    vals = [diffs[k] for k in keys]
    rng = random.Random(seed)
    n = len(keys)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    return {
        "delta":   point,
        "ci_low":  boots[int(0.025 * n_boot)],
        "ci_high": boots[min(n_boot - 1, int(0.975 * n_boot))],
        "n_clusters": n,
    }


def main() -> None:
    rows = json.loads(IN_FILE.read_text())
    families = sorted({r["attack_family"] for r in rows})
    rates    = sorted({r["rate"] for r in rows})
    seeds    = sorted({r["seed"] for r in rows})

    out: Dict[str, Any] = {
        "source": IN_FILE.name,
        "n_records": len(rows),
        "seeds": seeds,
        "n_bootstrap": N_BOOT,
        "note": ("content_tamper is the in-scope attack (digest mismatch); "
                 "genuine_substitution and withhold lie outside the "
                 "content-integrity guarantee, which claims neither "
                 "selection integrity nor completeness."),
        "by_family": {},
    }

    for fam in families:
        fam_rows = [r for r in rows if r["attack_family"] == fam]
        fam_out: Dict[str, Any] = {"by_rate": {}, "deltas": {}}

        for rate in rates:
            rr = [r for r in fam_rows if r["rate"] == rate]
            per_rate: Dict[str, Any] = {}

            for arm in ARMS:
                ar  = [r for r in rr if r["condition"] == arm]
                act = [r for r in ar if r["ground_truth"] in ACTIONABLE]
                urg = [r for r in ar if r["ground_truth"] == "urgent"]
                pois = [r for r in ar if r["poisoned"]]
                per_rate[arm] = {
                    "accuracy": (sum(r["decision"] == r["ground_truth"]
                                     for r in ar) / len(ar)) if ar else None,
                    "actionable": (sum(r["decision"] == r["ground_truth"]
                                       for r in act) / len(act)) if act else None,
                    "urgent_recall": (sum(r["decision"] == "urgent"
                                          for r in urg) / len(urg)) if urg else None,
                    # Mechanism-level: did the predicate ever fire, and did
                    # recovery ever run, on the poisoned scenarios?
                    "mean_tamper_alerts_poisoned": (
                        st.mean([r["tamper_alerts"] for r in pois])
                        if pois else 0.0),
                    "mean_peers_recovered_poisoned": (
                        st.mean([r["peers_recovered"] for r in pois])
                        if pois else 0.0),
                    "mean_n_peers_poisoned": (
                        st.mean([r["n_peers"] for r in pois])
                        if pois else 0.0),
                    "n_actionable": len(act),
                }
            if fam == "withhold":
                pw = [r for r in rr if r["poisoned"]]
                per_rate["mean_n_withheld"] = (
                    st.mean([r["n_withheld"] for r in pw]) if pw else 0.0)
            fam_out["by_rate"][f"{rate:.2f}"] = per_rate

        # Paired cluster-bootstrap differences on the actionable subset.
        for a, b in PAIRS:
            key = f"{a}-{b}"
            fam_out["deltas"][key] = {}
            for rate in rates:
                def per_scn(arm: str) -> Dict[str, List[int]]:
                    d = defaultdict(list)
                    for r in fam_rows:
                        if (r["rate"] == rate and r["condition"] == arm
                                and r["ground_truth"] in ACTIONABLE):
                            d[r["scenario_id"]].append(
                                int(r["decision"] == r["ground_truth"]))
                    return d
                fam_out["deltas"][key][f"{rate:.2f}"] = cluster_bootstrap(
                    per_scn(a), per_scn(b))
        out["by_family"][fam] = fam_out

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {OUT}\n")

    for fam in families:
        print(f"{fam}")
        print("  %-6s %8s %8s %8s %8s   %-24s %-24s"
              % ("rate", *ARMS, "T4-T2 [95% CI]", "T3-T2 [95% CI]"))
        for rate in rates:
            br = out["by_family"][fam]["by_rate"][f"{rate:.2f}"]
            d42 = out["by_family"][fam]["deltas"]["T4-T2"][f"{rate:.2f}"]
            d32 = out["by_family"][fam]["deltas"]["T3-T2"][f"{rate:.2f}"]
            print("  %-6.2f " % rate
                  + " ".join("%8.3f" % br[a]["actionable"] for a in ARMS)
                  + "   %+.3f [%+.3f,%+.3f]" % (d42["delta"], d42["ci_low"], d42["ci_high"])
                  + "  %+.3f [%+.3f,%+.3f]" % (d32["delta"], d32["ci_low"], d32["ci_high"]))
        print("  T4 detection on poisoned scenarios (alerts / recovered / peers seen):")
        for rate in rates:
            if rate == 0.0:
                continue
            t4 = out["by_family"][fam]["by_rate"][f"{rate:.2f}"]["T4"]
            print("    p=%.2f  %.2f / %.2f / %.2f"
                  % (rate, t4["mean_tamper_alerts_poisoned"],
                     t4["mean_peers_recovered_poisoned"],
                     t4["mean_n_peers_poisoned"]))
        print()


if __name__ == "__main__":
    main()
