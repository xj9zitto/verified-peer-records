"""
analyze_transfer_multiseed.py

Statistics for the multi-seed verification-transfer sweep.

Each attack seed redraws which scenarios are poisoned and which attack
each one receives, so seeds are independent attack realisations over a
fixed scenario corpus.  That structure has two consequences the analysis
must respect:

  * The same scenario recurs under every seed, so rows are not
    independent.  Bootstrap resampling is therefore done over
    *scenarios* (a cluster bootstrap), carrying all of a scenario's
    seeds with it.  Resampling rows instead would treat 20 correlated
    observations as 20 independent ones and understate the interval.

  * Variation across seeds is itself informative: it is the sampling
    distribution of the attack draw, and is reported as a between-seed
    standard deviation alongside the pooled estimate.

Output: results/transfer_multiseed_analysis.json
"""

from __future__ import annotations
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

RESULTS = Path("results")
import sys as _sys
IN_FILE = RESULTS / (_sys.argv[1] if len(_sys.argv) > 1
                     else "v14_verification_transfer_multiseed.json")
OUT     = RESULTS / (_sys.argv[2] if len(_sys.argv) > 2
                     else "transfer_multiseed_analysis.json")

SEED       = 42
N_BOOT     = 10000
ACTIONABLE = ("replace", "urgent")
SEVERITY   = {"safe": 0, "replace": 1, "urgent": 2}
ARMS       = ["T0", "T2", "T3", "T4"]

PAIRS = [("T4", "T2"), ("T4", "T3"), ("T4", "T0"), ("T3", "T2"), ("T2", "T0")]


def cluster_bootstrap(
    per_scenario: Dict[str, List[int]],
    per_scenario_b: Dict[str, List[int]],
    n_boot: int = N_BOOT, seed: int = SEED,
) -> Dict[str, float]:
    """
    CI for mean(a) - mean(b), resampling scenarios (clusters).  Each
    cluster contributes all of its seed-level observations.
    """
    keys = sorted(set(per_scenario) & set(per_scenario_b))
    # Per-scenario mean difference, then average over scenarios.
    diffs = {k: (st.mean(per_scenario[k]) - st.mean(per_scenario_b[k]))
             for k in keys}
    point = st.mean(diffs.values()) if keys else 0.0

    rng = random.Random(seed)
    n = len(keys)
    boots = []
    vals = [diffs[k] for k in keys]
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    return {
        "delta": point,
        "ci_low": boots[int(0.025 * n_boot)],
        "ci_high": boots[min(n_boot - 1, int(0.975 * n_boot))],
        "n_clusters": n,
    }


def main() -> None:
    recs = json.loads(IN_FILE.read_text())
    seeds = sorted({r["seed"] for r in recs})
    rates = sorted({r["rate"] for r in recs})
    print("records %d   seeds %d   rates %s" % (len(recs), len(seeds), rates))

    out: Dict[str, Any] = {
        "n_records": len(recs), "seeds": seeds, "rates": rates,
        "n_bootstrap": N_BOOT,
        "note": ("cluster bootstrap over scenarios; seeds are independent "
                 "attack realisations over a fixed scenario corpus"),
        "by_arm_rate": {}, "paired_deltas": {},
    }

    # -- pooled metrics + between-seed spread ------------------
    print("\n%-4s %-5s %8s %8s %10s %8s %9s"
          % ("arm", "rate", "act_acc", "sd_seed", "urg_recall",
             "under", "n_act"))
    for arm in ARMS:
        for rate in rates:
            grp = [r for r in recs
                   if r["condition"] == arm and r["rate"] == rate]
            act = [r for r in grp if r["ground_truth"] in ACTIONABLE]
            if not act:
                continue
            acc = sum(1 for r in act
                      if r["decision"] == r["ground_truth"]) / len(act)

            # per-seed accuracy -> between-seed SD
            per_seed = []
            for s in seeds:
                a = [r for r in act if r["seed"] == s]
                if a:
                    per_seed.append(
                        sum(1 for r in a
                            if r["decision"] == r["ground_truth"]) / len(a))
            sd = st.stdev(per_seed) if len(per_seed) > 1 else 0.0

            urg = [r for r in grp if r["ground_truth"] == "urgent"]
            urg_rec = (sum(1 for r in urg if r["decision"] == "urgent")
                       / len(urg)) if urg else None
            under = sum(1 for r in grp
                        if SEVERITY.get(r["decision"], 0)
                        < SEVERITY[r["ground_truth"]]) / len(grp)

            out["by_arm_rate"]["%s@%.2f" % (arm, rate)] = {
                "actionable_accuracy": acc,
                "between_seed_sd": sd,
                "per_seed_accuracy": per_seed,
                "urgent_recall": urg_rec,
                "under_triage_rate": under,
                "n_actionable": len(act),
            }
            print("%-4s %-5.2f %8.3f %8.3f %10.3f %8.3f %9d"
                  % (arm, rate, acc, sd, urg_rec, under, len(act)))
        print()

    # -- cluster-bootstrap paired deltas -----------------------
    print("Cluster bootstrap over scenarios, actionable subset:")
    print("%-10s %-5s %9s  %-22s" % ("pair", "rate", "delta", "95% CI"))
    for a, b in PAIRS:
        for rate in rates:
            pa: Dict[str, List[int]] = defaultdict(list)
            pb: Dict[str, List[int]] = defaultdict(list)
            for r in recs:
                if r["rate"] != rate or r["ground_truth"] not in ACTIONABLE:
                    continue
                ok = 1 if r["decision"] == r["ground_truth"] else 0
                if r["condition"] == a:
                    pa[r["scenario_id"]].append(ok)
                elif r["condition"] == b:
                    pb[r["scenario_id"]].append(ok)
            if not pa or not pb:
                continue
            d = cluster_bootstrap(pa, pb)
            key = "%s-%s@%.2f" % (a, b, rate)
            out["paired_deltas"][key] = d
            sig = "" if (d["ci_low"] <= 0 <= d["ci_high"]) else "  *"
            print("%-10s %-5.2f %+9.4f  [%+.4f, %+.4f]%s"
                  % ("%s-%s" % (a, b), rate, d["delta"],
                     d["ci_low"], d["ci_high"], sig))
        print()

    OUT.write_text(json.dumps(out, indent=2))
    print("Saved -> %s" % OUT)


if __name__ == "__main__":
    main()
