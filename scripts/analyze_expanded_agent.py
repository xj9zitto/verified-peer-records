"""
analyze_expanded_agent.py

Actionable-subset analysis of the independent 490-scenario agent run.

The set carries 175 actionable cases against the original 79, which is
enough for a single attack draw to resolve the C4-versus-C2 contrast
that required pooling on the smaller set.  Differences are paired on
scenario and bootstrapped over scenarios.

Output: results/expanded_agent_analysis.json
"""
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

RES = Path("results/v14_poisoning_expanded_results.json")
ACT = ("replace", "urgent")
SEV = {"safe": 0, "replace": 1, "urgent": 2}
N_BOOT = 10000

recs = [r for r in json.loads(RES.read_text())
        if r.get("classification") != "error"]
rates = sorted({r["rate"] for r in recs})
print("records %d  rates %s" % (len(recs), rates))
print("class counts:", dict(Counter(r["ground_truth"] for r in recs
                                    if r["condition"] == "C2"
                                    and r["rate"] == 0.0)))

print("\n%-5s %-5s %8s %9s %9s %9s %7s"
      % ("cond", "rate", "acc", "act_acc", "urg_rec", "under", "n_act"))
for cond in ("C2", "C4"):
    for rate in rates:
        g = [r for r in recs if r["condition"] == cond and r["rate"] == rate]
        a = [r for r in g if r["ground_truth"] in ACT]
        u = [r for r in g if r["ground_truth"] == "urgent"]
        acc = sum(1 for r in g if r["decision"] == r["ground_truth"]) / len(g)
        aacc = sum(1 for r in a if r["decision"] == r["ground_truth"]) / len(a)
        urec = sum(1 for r in u if r["decision"] == "urgent") / len(u)
        und = sum(1 for r in g
                  if SEV.get(r["decision"], 0) < SEV[r["ground_truth"]]) / len(g)
        print("%-5s %-5.2f %8.3f %9.3f %9.3f %9.3f %7d"
              % (cond, rate, acc, aacc, urec, und, len(a)))
    print()


def boot(pairs, seed=42):
    diffs = [x - y for x, y in pairs]
    n = len(diffs)
    pt = st.mean(diffs)
    rng = random.Random(seed)
    bs = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(N_BOOT))
    return pt, bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT)], n


report = {"n_records": len(recs), "rates": rates, "by_condition_rate": {},
          "paired": {}}
for cond in ("C2", "C4"):
    for rate in rates:
        g = [r for r in recs if r["condition"] == cond and r["rate"] == rate]
        a = [r for r in g if r["ground_truth"] in ACT]
        u = [r for r in g if r["ground_truth"] == "urgent"]
        report["by_condition_rate"]["%s@%.2f" % (cond, rate)] = {
            "accuracy": sum(1 for r in g
                            if r["decision"] == r["ground_truth"]) / len(g),
            "actionable_accuracy": sum(
                1 for r in a if r["decision"] == r["ground_truth"]) / len(a),
            "urgent_recall": sum(1 for r in u
                                 if r["decision"] == "urgent") / len(u),
            "under_triage_rate": sum(
                1 for r in g
                if SEV.get(r["decision"], 0) < SEV[r["ground_truth"]]) / len(g),
            "n_actionable": len(a),
        }

print("Paired C4 minus C2, actionable subset:")
for rate in rates:
    idx = {}
    for r in recs:
        if r["rate"] != rate or r["ground_truth"] not in ACT:
            continue
        idx.setdefault(r["scenario_id"], {})[r["condition"]] = (
            1 if r["decision"] == r["ground_truth"] else 0)
    pairs = [(v["C4"], v["C2"]) for v in idx.values()
             if "C4" in v and "C2" in v]
    d, lo, hi, n = boot(pairs)
    sig = "" if lo <= 0 <= hi else "  *"
    print("  p=%.2f  %+.4f  [%+.4f, %+.4f]  n=%d%s" % (rate, d, lo, hi, n, sig))
    report["paired"]["%.2f" % rate] = {"delta": d, "ci_low": lo,
                                       "ci_high": hi, "n_pairs": n}

OUT = Path("results/expanded_agent_analysis.json")
OUT.write_text(json.dumps(report, indent=2))
print("")
print("Saved -> %s" % OUT)
