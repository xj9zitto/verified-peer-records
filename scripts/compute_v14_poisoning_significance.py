"""
compute_v14_poisoning_significance.py

McNemar's paired test, C3 vs each baseline at every corruption rate,
restricted to the **actionable subset** (GT in {replace, urgent})
where the attack actually matters.

Output: results/v14_poisoning_significance.json
"""

import json
import math
from pathlib import Path
from scipy.stats import binomtest, chi2

RESULTS_DIR = Path("results")
RAW_FILE    = RESULTS_DIR / "v14_poisoning_results.json"
OUT_FILE    = RESULTS_DIR / "v14_poisoning_significance.json"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return {"p_value": 1.0, "chi2": None, "b": 0, "c": 0,
                "n_discordant": 0, "exact": True}
    if n < 25:
        p = binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
        return {"p_value": float(p), "chi2": None, "b": b, "c": c,
                "n_discordant": n, "exact": True}
    chi = (abs(b - c) - 1) ** 2 / n
    p = float(1 - chi2.cdf(chi, df=1))
    return {"p_value": p, "chi2": float(chi), "b": b, "c": c,
            "n_discordant": n, "exact": False}


def main():
    recs = json.loads(RAW_FILE.read_text())

    # Index: (rate, scenario_id, condition) -> correct(0/1) for actionable subset only
    idx = {}
    for r in recs:
        if r.get("classification") == "error":
            continue
        if r["ground_truth"] not in ("replace", "urgent"):
            continue
        key = (r["rate"], r["scenario_id"], r["condition"])
        idx[key] = 1 if r["decision"] == r["ground_truth"] else 0

    rates = sorted({r["rate"] for r in recs})
    sids  = sorted({r["scenario_id"] for r in recs
                    if r["ground_truth"] in ("replace", "urgent")})

    summary = {"alpha": 0.05, "actionable_subset_size": len(sids),
               "by_rate": {}}

    print("\nMcNemar's paired test on actionable subset (GT in {replace,urgent})")
    print(f"{'rate':>5}  {'pair':>9}  {'acc_C3':>7}  {'acc_o':>7}  "
          f"{'b':>4}  {'c':>4}  {'p':>10}  sig?")

    for rate in rates:
        rate_summary = {}
        for cond_b in ["C0", "C1", "C2"]:
            b, c = 0, 0
            acc_a_correct = acc_a_total = 0
            acc_b_correct = acc_b_total = 0
            for sid in sids:
                a = idx.get((rate, sid, "C3"))
                d = idx.get((rate, sid, cond_b))
                if a is None or d is None:
                    continue
                acc_a_total += 1
                acc_b_total += 1
                acc_a_correct += a
                acc_b_correct += d
                if a == 1 and d == 0:
                    b += 1
                elif a == 0 and d == 1:
                    c += 1
            res = mcnemar(b, c)
            res["acc_C3"]    = (acc_a_correct / acc_a_total
                                if acc_a_total else None)
            res["acc_other"] = (acc_b_correct / acc_b_total
                                if acc_b_total else None)
            res["delta_C3_minus_other"] = (
                res["acc_C3"] - res["acc_other"]
                if res["acc_C3"] is not None and res["acc_other"] is not None
                else None
            )
            sig = ("**" if res["p_value"] < 0.0125
                   else "*" if res["p_value"] < 0.05 else "")
            print(f"{rate:>5.2f}  C3 vs {cond_b}  "
                  f"{res['acc_C3']:>7.3f}  {res['acc_other']:>7.3f}  "
                  f"{b:>4d}  {c:>4d}  {res['p_value']:>10.4g}  {sig}")
            rate_summary[f"C3_vs_{cond_b}"] = res
        summary["by_rate"][f"{rate:.2f}"] = rate_summary

    OUT_FILE.write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
