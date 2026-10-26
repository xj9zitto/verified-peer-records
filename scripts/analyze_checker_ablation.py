"""
analyze_checker_ablation.py

Ablation of the physics-consistency gate, as requested in review.

The gated run records both the repaired action (``decision``) and the
model's unrepaired action (``decision_raw``), so the with-gate and
without-gate arms come from a single run and are paired on identical
model outputs.  That is a stronger comparison than re-running with the
gate disabled, which would confound the gate's effect with the model's
own run-to-run variance (measured separately at roughly 3% of
decisions).

Reports: how often the gate fires, which violations it catches, which
fallback it applies, and what it costs or buys on accuracy, urgent
recall, and under-triage.

Output: results/checker_ablation.json
"""

from __future__ import annotations
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

RESULTS = Path("results")
IN_FILE = RESULTS / "v14_poisoning_c4_gated_results.json"
OUT = RESULTS / "checker_ablation.json"

SEED = 42
N_BOOT = 10000
ACTIONABLE = ("replace", "urgent")
SEVERITY = {"safe": 0, "replace": 1, "urgent": 2}


def paired_bootstrap(pairs: List[tuple], n_boot: int = N_BOOT,
                     seed: int = SEED) -> Dict[str, float]:
    """CI for mean(a) - mean(b) over paired (a, b) correctness."""
    diffs = [a - b for a, b in pairs]
    if not diffs:
        return {}
    point = st.mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return {
        "delta": point,
        "ci_low": boots[int(0.025 * n_boot)],
        "ci_high": boots[min(n_boot - 1, int(0.975 * n_boot))],
        "n_pairs": n,
    }


def arm_metrics(recs: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    n = len(recs)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for r in recs if r.get(field) == r["ground_truth"])
    act = [r for r in recs if r["ground_truth"] in ACTIONABLE]
    act_correct = sum(1 for r in act if r.get(field) == r["ground_truth"])
    urg = [r for r in recs if r["ground_truth"] == "urgent"]
    urg_rec = (sum(1 for r in urg if r.get(field) == "urgent") / len(urg)
               if urg else None)
    under = sum(1 for r in recs
                if SEVERITY.get(r.get(field), 0)
                < SEVERITY[r["ground_truth"]]) / n
    return {
        "n": n,
        "accuracy": correct / n,
        "actionable_accuracy": (act_correct / len(act)) if act else None,
        "actionable_n": len(act),
        "urgent_recall": urg_rec,
        "under_triage_rate": under,
    }


def main() -> None:
    if not IN_FILE.exists():
        raise SystemExit("missing %s -- run the gated experiment first"
                         % IN_FILE)
    recs = json.loads(IN_FILE.read_text())
    recs = [r for r in recs if r.get("classification") != "error"]
    rates = sorted({r["rate"] for r in recs})
    print("records %d   rates %s" % (len(recs), rates))

    # -- how often does the gate fire, and on what? ------------
    viol = Counter()
    fallback = Counter()
    fired = 0
    for r in recs:
        vs = r.get("violations") or []
        if vs:
            fired += 1
        for v in vs:
            viol[v] += 1
        fallback[r.get("fallback_applied", "none")] += 1

    print("\ngate fired on %d/%d decisions (%.1f%%)"
          % (fired, len(recs), 100.0 * fired / len(recs)))
    print("violations:")
    for k, v in viol.most_common():
        print("  %-32s %5d" % (k, v))
    print("fallback applied:")
    for k, v in fallback.most_common():
        print("  %-32s %5d (%.1f%%)" % (k, v, 100.0 * v / len(recs)))

    out: Dict[str, Any] = {
        "n_records": len(recs),
        "gate_fired": fired,
        "gate_fire_rate": fired / len(recs),
        "violations": dict(viol),
        "fallback_applied": dict(fallback),
        "by_rate": {},
        "paired": {},
    }

    # -- paired with/without comparison ------------------------
    print("\n%-6s %-9s %8s %9s %9s %9s"
          % ("rate", "arm", "acc", "act_acc", "urg_rec", "under"))
    for rate in rates + ["all"]:
        grp = (recs if rate == "all"
               else [r for r in recs if r["rate"] == rate])
        gated = arm_metrics(grp, "decision")
        raw = arm_metrics(grp, "decision_raw")
        key = "all" if rate == "all" else "%.2f" % rate
        out["by_rate"][key] = {"gated": gated, "ungated": raw}

        for name, m in (("gated", gated), ("ungated", raw)):
            print("%-6s %-9s %8.3f %9s %9s %9.3f"
                  % (key, name, m["accuracy"],
                     "%.3f" % m["actionable_accuracy"]
                     if m["actionable_accuracy"] is not None else "NA",
                     "%.3f" % m["urgent_recall"]
                     if m["urgent_recall"] is not None else "NA",
                     m["under_triage_rate"]))

        # paired deltas, overall and actionable
        pairs_all = [(1 if r.get("decision") == r["ground_truth"] else 0,
                      1 if r.get("decision_raw") == r["ground_truth"] else 0)
                     for r in grp]
        pairs_act = [p for p, r in zip(pairs_all, grp)
                     if r["ground_truth"] in ACTIONABLE]
        out["paired"][key] = {
            "overall": paired_bootstrap(pairs_all),
            "actionable": paired_bootstrap(pairs_act),
        }
        print()

    print("Paired gate effect (gated minus ungated):")
    for key, d in out["paired"].items():
        for subset in ("overall", "actionable"):
            v = d[subset]
            if not v:
                continue
            sig = "" if v["ci_low"] <= 0 <= v["ci_high"] else "  *"
            print("  %-5s %-11s %+.4f  [%+.4f, %+.4f]  n=%d%s"
                  % (key, subset, v["delta"], v["ci_low"], v["ci_high"],
                     v["n_pairs"], sig))

    OUT.write_text(json.dumps(out, indent=2))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
