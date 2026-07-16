"""
compute_poisoning_significance.py

McNemar's test (paired) for C3 vs each other condition at every corruption
rate of the input-poisoning experiment.  Saves results to
results/poisoning_significance.json.

Usage:
    python scripts/compute_poisoning_significance.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest, chi2

RESULTS_DIR = Path("results")
RAW_FILE    = RESULTS_DIR / "input_poisoning_results.json"
OUT_FILE    = RESULTS_DIR / "poisoning_significance.json"

_GT_MAP = {
    "r1": "reroute", "r2": "reroute", "r3": "reroute", "r4": "reroute",
    "reroute": "reroute",
    "no_safe_route_available": "no_safe_route", "no_safe_route": "no_safe_route",
    "reorder": "reorder", "reorder_degradation_flag": "reorder",
    "replenish": "reorder",
    "k1": "carrier", "k2": "carrier", "k3": "carrier",
    "use k1": "carrier", "use k2": "carrier", "use k3": "carrier",
    "carrier": "carrier", "hold": "hold",
    "uncertain": "uncertain", "error": "error",
}

def _norm(s):
    return _GT_MAP.get(str(s).lower().strip(), str(s).lower().strip())


def mcnemar(b: int, c: int):
    """
    Two-sided McNemar's test on paired binary outcomes.
    b = #pairs where A correct, B incorrect
    c = #pairs where B correct, A incorrect
    Uses exact binomial when b+c < 25, otherwise chi^2 with continuity.
    Returns dict with p_value, statistic, exact, n_discordant.
    """
    n = b + c
    if n == 0:
        return {"p_value": 1.0, "chi2": None, "b": 0, "c": 0,
                "n_discordant": 0, "exact": True}
    if n < 25:
        # exact two-sided binomial
        p = binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
        return {"p_value": float(p), "chi2": None, "b": b, "c": c,
                "n_discordant": n, "exact": True}
    chi = (abs(b - c) - 1) ** 2 / n
    p = float(1 - chi2.cdf(chi, df=1))
    return {"p_value": p, "chi2": float(chi), "b": b, "c": c,
            "n_discordant": n, "exact": False}


def main():
    if not RAW_FILE.exists():
        print(f"Missing {RAW_FILE}. Run the experiment first.")
        return
    records = json.loads(RAW_FILE.read_text())

    # Index by (rate, scenario_id, condition) → correct(0/1)
    idx = {}
    for r in records:
        if r.get("classification") == "error":
            continue
        key = (r["rate"], r["scenario_id"], r["condition"])
        idx[key] = 1 if _norm(r["ground_truth"]) == _norm(r["decision"]) else 0

    rates = sorted({r["rate"] for r in records})
    sids  = sorted({r["scenario_id"] for r in records})

    summary = {"alpha": 0.05, "rates": rates, "n_scenarios": len(sids),
               "by_rate": {}}

    print("\nMcNemar's paired test — C3 vs other conditions, by rate")
    print(f"{'rate':>5}  {'pair':>9}  {'acc_a':>6}  {'acc_b':>6}  "
          f"{'b':>4}  {'c':>4}  {'p':>10}  sig?")

    for rate in rates:
        rate_summary = {}
        # accuracy per condition at this rate
        acc_per_cond = {}
        for cond in ["C0", "C1", "C2", "C3"]:
            vals = [idx.get((rate, sid, cond)) for sid in sids]
            vals = [v for v in vals if v is not None]
            acc_per_cond[cond] = sum(vals) / len(vals) if vals else None

        for cond_b in ["C0", "C1", "C2"]:
            b, c = 0, 0
            for sid in sids:
                a = idx.get((rate, sid, "C3"))
                d = idx.get((rate, sid, cond_b))
                if a is None or d is None:
                    continue
                if a == 1 and d == 0:
                    b += 1
                elif a == 0 and d == 1:
                    c += 1
            res = mcnemar(b, c)
            res["acc_C3"]   = acc_per_cond["C3"]
            res["acc_other"] = acc_per_cond[cond_b]
            res["delta_C3_minus_other"] = (
                acc_per_cond["C3"] - acc_per_cond[cond_b]
                if acc_per_cond["C3"] is not None and acc_per_cond[cond_b] is not None
                else None
            )
            sig = "**" if res["p_value"] < 0.0125 else ("*" if res["p_value"] < 0.05 else "")
            print(f"{rate:>5.2f}  C3 vs {cond_b}  "
                  f"{res['acc_C3']:>6.3f}  {res['acc_other']:>6.3f}  "
                  f"{b:>4d}  {c:>4d}  {res['p_value']:>10.4g}  {sig}")
            rate_summary[f"C3_vs_{cond_b}"] = res

        summary["by_rate"][f"{rate:.2f}"] = rate_summary

    OUT_FILE.write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
