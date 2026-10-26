"""
compute_mcnemar_corrected.py

Paired McNemar tests on the actionable subset, recomputed with the
deployable C3 rather than the oracle variant.

The published table was computed before C3's recovery was made
deployable, so it compared an oracle that also repaired the falsified
observable header.  Those tests overstate C3 and are not comparable
with C2 or C4, which is the objection the review raised.  This script
recomputes every contrast from the deployable run and applies a Holm
correction across the family of tests reported in the table.

Output: results/mcnemar_corrected.json
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scipy.stats import binomtest, chi2

RESULTS = Path("results")
OUT = RESULTS / "mcnemar_corrected.json"
ACTIONABLE = ("replace", "urgent")

SOURCES = [
    ("v14_poisoning_results.json", None),                    # C0,C1,C2,C3(oracle)
    ("v14_poisoning_c4_results.json", None),                 # C4
    ("v14_poisoning_c3_deployable_results.json", "C3"),      # deployable C3
]


def mcnemar(b: int, c: int) -> Dict[str, Any]:
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p": 1.0, "exact": True}
    if n < 25:
        p = float(binomtest(min(b, c), n=n, p=0.5,
                            alternative="two-sided").pvalue)
        return {"b": b, "c": c, "n_discordant": n, "p": p, "exact": True}
    chi = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "n_discordant": n,
            "p": float(1 - chi2.cdf(chi, df=1)), "exact": False}


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    # condition -> rate -> scenario -> correct
    idx: Dict[str, Dict[float, Dict[str, int]]] = {}
    acc: Dict[Tuple[str, float], float] = {}

    for fname, relabel in SOURCES:
        p = RESULTS / fname
        if not p.exists():
            print("  skip (missing): %s" % fname)
            continue
        for r in json.loads(p.read_text()):
            if r.get("classification") == "error":
                continue
            if r["ground_truth"] not in ACTIONABLE:
                continue
            cond = r["condition"]
            if relabel and cond == relabel:
                cond = "C3"          # deployable replaces the oracle rows
            elif not relabel and cond == "C3":
                cond = "C3o"         # keep the oracle separately
            idx.setdefault(cond, {}).setdefault(r["rate"], {})[
                r["scenario_id"]] = 1 if r["decision"] == r["ground_truth"] else 0

    for cond, byrate in idx.items():
        for rate, m in byrate.items():
            acc[(cond, rate)] = sum(m.values()) / len(m)

    rates = sorted({rt for by in idx.values() for rt in by})
    print("conditions: %s" % sorted(idx))
    print("\n%-6s %-9s %-9s %6s %6s %6s %10s %10s"
          % ("rate", "cond", "vs", "b", "c", "n", "p_raw", "delta_pp"))

    tests: List[Dict[str, Any]] = []
    for rate in rates:
        for cond, comp in (("C4", "C0"), ("C4", "C1"), ("C4", "C2"),
                           ("C4", "C3"), ("C3", "C0"), ("C3", "C2")):
            a = idx.get(cond, {}).get(rate)
            b_ = idx.get(comp, {}).get(rate)
            if not a or not b_:
                continue
            keys = sorted(set(a) & set(b_))
            b = sum(1 for k in keys if a[k] == 1 and b_[k] == 0)
            c = sum(1 for k in keys if a[k] == 0 and b_[k] == 1)
            res = mcnemar(b, c)
            delta = 100.0 * (acc[(cond, rate)] - acc[(comp, rate)])
            res.update({"rate": rate, "condition": cond, "comparator": comp,
                        "delta_pp": delta, "n_pairs": len(keys)})
            tests.append(res)
            print("%-6.2f %-9s %-9s %6d %6d %6d %10.4f %+10.1f"
                  % (rate, cond, comp, b, c, res["n_discordant"],
                     res["p"], delta))

    adj = holm([t["p"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_holm"] = a
        t["significant_holm"] = a < 0.05

    print("\nSurviving Holm correction across %d tests:" % len(tests))
    any_sig = False
    for t in tests:
        if t["significant_holm"]:
            any_sig = True
            print("  p=%.2f  %s vs %s  %+.1f pp  raw %.4f  Holm %.4f"
                  % (t["rate"], t["condition"], t["comparator"],
                     t["delta_pp"], t["p"], t["p_holm"]))
    if not any_sig:
        print("  none")

    OUT.write_text(json.dumps(
        {"accuracies": {"%s@%.2f" % k: v for k, v in acc.items()},
         "tests": tests}, indent=2))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
