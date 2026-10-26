"""
analyze_seed_campaign.py

Pools the agent conditions across independent attack draws and
quantifies how much of the remaining variance is the model rather than
the adversary, addressing the statistical-power comment.

Each attack seed redraws which scenarios are poisoned and which attack
each receives, so seeds are independent realisations of the adversary
over a fixed scenario corpus.  Scenarios recur across seeds, so the
bootstrap resamples scenarios (clusters) and carries all of a
scenario's observations together; resampling rows would treat repeated
measurements of the same scenario as independent.

Two questions are answered separately:

  power        pooling seeds, does C4 beat C2 on the actionable subset?
  determinism  re-running C4 under an identical attack draw, how many
               decisions change?  This bounds how large a difference a
               single-run comparison can resolve at all.

Output: results/seed_campaign_analysis.json
"""

from __future__ import annotations
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

RESULTS = Path("results")
OUT = RESULTS / "seed_campaign_analysis.json"

SEED = 42
N_BOOT = 10000
ACTIONABLE = ("replace", "urgent")

# (file, default attack seed when the field predates the flag)
SOURCES = [
    ("v14_poisoning_results.json", 42),
    ("v14_poisoning_c4_results.json", 42),
    ("v14_poisoning_seed43_results.json", 43),
    ("v14_poisoning_seed44_results.json", 44),
]
REPEAT_FILE = RESULTS / "v14_poisoning_c4_repeat_results.json"
BASE_C4_FILE = RESULTS / "v14_poisoning_c4_results.json"


def cluster_bootstrap(per_scen_a, per_scen_b, n_boot=N_BOOT, seed=SEED):
    keys = sorted(set(per_scen_a) & set(per_scen_b))
    if not keys:
        return {}
    vals = [st.mean(per_scen_a[k]) - st.mean(per_scen_b[k]) for k in keys]
    point = st.mean(vals)
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return {
        "delta": point,
        "ci_low": boots[int(0.025 * n_boot)],
        "ci_high": boots[min(n_boot - 1, int(0.975 * n_boot))],
        "n_clusters": n,
    }


def main() -> None:
    recs: List[Dict[str, Any]] = []
    loaded = []
    for fname, default_seed in SOURCES:
        p = RESULTS / fname
        if not p.exists():
            print("  skip (missing): %s" % fname)
            continue
        data = json.loads(p.read_text())
        for r in data:
            r.setdefault("attack_seed", default_seed)
        recs.extend(data)
        loaded.append("%s (%d)" % (fname, len(data)))
    print("Loaded: " + ", ".join(loaded))

    recs = [r for r in recs
            if r.get("classification") != "error"
            and r["condition"] in ("C2", "C4")]
    seeds = sorted({r["attack_seed"] for r in recs})
    rates = sorted({r["rate"] for r in recs})
    print("conditions C2/C4 -- attack seeds %s, rates %s" % (seeds, rates))

    out: Dict[str, Any] = {
        "sources": loaded, "seeds": seeds, "n_bootstrap": N_BOOT,
        "per_seed": {}, "pooled": {}, "determinism": {},
    }

    # -- per-seed actionable accuracy ---------------------------
    print("\n%-6s %-5s %8s %8s %9s" % ("seed", "rate", "C2", "C4", "n_act"))
    for sd in seeds:
        for rate in rates:
            row = {}
            n_act = 0
            for cond in ("C2", "C4"):
                g = [r for r in recs
                     if r["attack_seed"] == sd and r["rate"] == rate
                     and r["condition"] == cond
                     and r["ground_truth"] in ACTIONABLE]
                if not g:
                    continue
                n_act = len(g)
                row[cond] = sum(1 for r in g
                                if r["decision"] == r["ground_truth"]) / len(g)
            if len(row) == 2:
                out["per_seed"]["%d@%.2f" % (sd, rate)] = {
                    **row, "n_actionable": n_act}
                print("%-6d %-5.2f %8.3f %8.3f %9d"
                      % (sd, rate, row["C2"], row["C4"], n_act))

    # -- pooled cluster bootstrap -------------------------------
    print("\nPooled over %d attack seeds, actionable subset "
          "(C4 minus C2):" % len(seeds))
    for rate in rates:
        pa, pb = defaultdict(list), defaultdict(list)
        for r in recs:
            if r["rate"] != rate or r["ground_truth"] not in ACTIONABLE:
                continue
            ok = 1 if r["decision"] == r["ground_truth"] else 0
            if r["condition"] == "C4":
                pa[r["scenario_id"]].append(ok)
            elif r["condition"] == "C2":
                pb[r["scenario_id"]].append(ok)
        d = cluster_bootstrap(pa, pb)
        if not d:
            continue
        n_obs = sum(len(v) for v in pa.values())
        d["n_observations"] = n_obs
        out["pooled"]["%.2f" % rate] = d
        sig = "" if d["ci_low"] <= 0 <= d["ci_high"] else "  *"
        print("  p=%.2f  %+.4f  [%+.4f, %+.4f]  clusters=%d obs=%d%s"
              % (rate, d["delta"], d["ci_low"], d["ci_high"],
                 d["n_clusters"], n_obs, sig))

    # -- determinism: repeat under an identical attack draw -----
    if REPEAT_FILE.exists() and BASE_C4_FILE.exists():
        a = {(r["scenario_id"], r["rate"]): r
             for r in json.loads(BASE_C4_FILE.read_text())}
        b = {(r["scenario_id"], r["rate"]): r
             for r in json.loads(REPEAT_FILE.read_text())}
        keys = sorted(set(a) & set(b))
        flips = [k for k in keys if a[k]["decision"] != b[k]["decision"]]
        acc_a = sum(1 for k in keys
                    if a[k]["decision"] == a[k]["ground_truth"]) / len(keys)
        acc_b = sum(1 for k in keys
                    if b[k]["decision"] == b[k]["ground_truth"]) / len(keys)
        out["determinism"] = {
            "n_compared": len(keys),
            "n_flipped": len(flips),
            "flip_rate": len(flips) / len(keys),
            "accuracy_run1": acc_a,
            "accuracy_run2": acc_b,
            "accuracy_delta": acc_b - acc_a,
        }
        print("\nDeterminism (C4 re-run, identical attack draw):")
        print("  %d/%d decisions changed (%.1f%%); accuracy %.3f vs %.3f"
              % (len(flips), len(keys), 100.0 * len(flips) / len(keys),
                 acc_a, acc_b))
        print("  Individual decisions churn at %.1f%%, while aggregate\n"
              "  accuracy moves only %.1f pp: flips largely cancel, so the\n"
              "  flip rate bounds paired per-decision analyses, not the\n"
              "  reproducibility of the aggregate."
              % (100.0 * len(flips) / len(keys),
                 100.0 * abs(acc_b - acc_a)))
    else:
        print("\n(repeat run not present yet; determinism section skipped)")

    OUT.write_text(json.dumps(out, indent=2))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
