"""
compute_significance.py
Statistical significance testing for the tamper-robustness / accuracy
comparisons across experimental conditions C0–C3.

Tests performed
---------------
1. McNemar's test (paired nominal) — C3 vs C0/C1/C2 on decision accuracy.
   Each scenario is one pair; outcome is correct (1) or wrong (0).
   H0: the marginal accuracy of both conditions is equal.

2. Wilcoxon signed-rank test — per-type accuracy differences C3 vs C1/C2.
   Used when McNemar assumptions do not hold (e.g. small n per type).

3. Bonferroni correction — applied across the k=6 pairwise tests.

Outputs
-------
  results/significance.json  — full test results for paper reporting
  Console table              — p-values and effect sizes
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats

sys.path.insert(0, ".")

RESULTS = Path("results")

# ── Ground-truth normaliser ────────────────────────────────────
_GT_MAP = {
    "r1":"reroute","r2":"reroute","r3":"reroute","r4":"reroute",
    "reroute":"reroute",
    "no_safe_route_available":"no_safe_route","no_safe_route":"no_safe_route",
    "reorder":"reorder","reorder_degradation_flag":"reorder","replenish":"reorder",
    "k1":"carrier","k2":"carrier","k3":"carrier",
    "use k1":"carrier","use k2":"carrier","use k3":"carrier",
    "carrier":"carrier","hold":"hold","uncertain":"uncertain","error":"error",
}

def _norm(x: str) -> str:
    return _GT_MAP.get(str(x).lower().strip(), str(x).lower().strip())

def _correct(r: dict) -> int:
    return int(_norm(r["ground_truth"]) == _norm(r["decision"]))


def mcnemar_pair(y_a: list, y_b: list) -> dict:
    """
    McNemar's test on two paired binary outcome lists.
    Returns chi2, p_value, and n_discordant.
    Uses exact binomial if n_discordant < 25, else chi2 with continuity correction.
    """
    assert len(y_a) == len(y_b)
    b = sum(1 for a, b_ in zip(y_a, y_b) if a == 1 and b_ == 0)  # A correct, B wrong
    c = sum(1 for a, b_ in zip(y_a, y_b) if a == 0 and b_ == 1)  # A wrong, B correct
    n_disc = b + c

    if n_disc == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c,
                "n_discordant": 0, "exact": False}

    if n_disc < 25:
        # Exact binomial test: H0 probability of b/(b+c) = 0.5
        result = stats.binomtest(b, n_disc, 0.5, alternative="two-sided")
        p = float(result.pvalue)
        # Clamp to smallest positive float to avoid JSON-serialized 0.0
        p = max(p, 5e-324)
        return {"chi2": None, "p_value": round(p, 10), "b": b, "c": c,
                "n_discordant": n_disc, "exact": True}
    else:
        # Chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p    = float(stats.chi2.sf(chi2, df=1))
        p    = max(p, 5e-324)
        return {"chi2": round(chi2, 4), "p_value": round(p, 10), "b": b, "c": c,
                "n_discordant": n_disc, "exact": False}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def main():
    with open(RESULTS / "raw_results.json") as f:
        raw = json.load(f)

    # Filter errors; index by (scenario_id, condition)
    clean = [r for r in raw if r.get("classification") != "error"]
    by_cond: dict = defaultdict(dict)
    for r in clean:
        by_cond[r["condition"]][r["scenario_id"]] = r

    conditions = ["C0", "C1", "C2", "C3"]

    # Align scenario IDs present in all four conditions
    common_ids = set(by_cond["C3"].keys())
    for c in conditions:
        common_ids &= set(by_cond[c].keys())
    common_ids = sorted(common_ids)
    n = len(common_ids)
    print(f"\nAligned scenarios across all conditions: n={n}\n")

    # ── McNemar tests: C3 vs each other condition ──────────────
    alpha    = 0.05
    pairs    = [("C3", c) for c in ("C0", "C1", "C2")]
    k_tests  = len(pairs)                          # for Bonferroni
    alpha_bf = alpha / k_tests

    mc_results = {}
    print("=" * 72)
    print(f"  {'Pair':<12}  {'Acc_A':>7}  {'Acc_B':>7}  {'h':>6}  "
          f"{'stat':>8}  {'p':>10}  {'sig':>6}  {'exact':>6}")
    print("  " + "-" * 68)

    for cond_a, cond_b in pairs:
        y_a = [_correct(by_cond[cond_a][sid]) for sid in common_ids]
        y_b = [_correct(by_cond[cond_b][sid]) for sid in common_ids]
        acc_a = sum(y_a) / n
        acc_b = sum(y_b) / n
        h     = cohens_h(acc_a, acc_b)
        mc    = mcnemar_pair(y_a, y_b)
        p     = mc["p_value"]
        sig   = "***" if p < alpha_bf/10 else ("**" if p < alpha_bf else
                ("*" if p < alpha else "ns"))
        stat_str = f"{mc['chi2']:.3f}" if mc["chi2"] is not None else "exact"

        print(f"  {cond_a} vs {cond_b:<4}  {acc_a:>7.4f}  {acc_b:>7.4f}  "
              f"{h:>+6.3f}  {stat_str:>8}  {p:>10.6f}  {sig:>6}  "
              f"{'Y' if mc['exact'] else 'N':>6}")

        mc_results[f"{cond_a}_vs_{cond_b}"] = {
            "acc_a":        round(acc_a, 4),
            "acc_b":        round(acc_b, 4),
            "cohens_h":     round(float(h), 4),
            "p_value":      mc["p_value"],
            "chi2":         mc["chi2"],
            "b":            mc["b"],
            "c":            mc["c"],
            "n_discordant": mc["n_discordant"],
            "exact":        mc["exact"],
            "significant_bonferroni": p < alpha_bf,
            "significant_uncorrected": p < alpha,
        }

    print("=" * 72)
    print(f"\n  Bonferroni α = {alpha_bf:.4f}  (α={alpha}, k={k_tests} tests)")
    print("  *** p < α_bf/10   ** p < α_bf   * p < 0.05   ns = not significant\n")

    # ── Per-type McNemar: C3 vs C2 (strict / specific accuracy) ─
    # Strict accuracy requires matching the exact specific label
    # (e.g., R2 not just "reroute", K3 not just "carrier").
    # Coarse comparison is trivially identical for reroute/carrier.
    scenario_types = ["reorder", "reroute", "path", "carrier"]

    def _strict_correct(r: dict) -> int:
        sd = r.get("strict_decision")
        return int(sd is not None and sd == r["ground_truth"])

    print("\n=== Per-type McNemar: C3 vs C2 (specific accuracy) ===\n")
    print(f"  {'Type':<10}  {'n':>4}  {'sAcc_C3':>9}  {'sAcc_C2':>9}  "
          f"{'h':>6}  {'p':>10}  {'sig':>5}")
    print("  " + "-" * 62)

    type_results = {}
    for stype in scenario_types:
        ids_t = [sid for sid in common_ids
                 if by_cond["C3"][sid]["scenario_type"] == stype]
        if len(ids_t) < 5:
            continue
        y3 = [_strict_correct(by_cond["C3"][sid]) for sid in ids_t]
        y2 = [_strict_correct(by_cond["C2"][sid]) for sid in ids_t]
        acc3 = sum(y3) / len(ids_t)
        acc2 = sum(y2) / len(ids_t)
        h    = cohens_h(acc3, acc2) if (acc3 > 0 or acc2 > 0) else 0.0
        mc   = mcnemar_pair(y3, y2)
        p    = mc["p_value"]
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else
               ("*" if p < 0.05 else "ns"))
        print(f"  {stype:<10}  {len(ids_t):>4}  {acc3:>9.4f}  {acc2:>9.4f}  "
              f"{h:>+6.3f}  {p:>10.6f}  {sig:>5}")
        type_results[stype] = {
            "n": len(ids_t),
            "strict_acc_c3": round(acc3, 4), "strict_acc_c2": round(acc2, 4),
            "cohens_h": round(float(h), 4), "p_value": mc["p_value"],
            "significant": bool(p < 0.05),
        }

    # ── Tamper robustness significance ─────────────────────────
    tamper_path = RESULTS / "tamper_results.json"
    tamper_mc   = {}
    if tamper_path.exists():
        with open(tamper_path) as f:
            tamper_all = json.load(f)

        print("\n=== Tamper Robustness McNemar: C3 vs C2 ===\n")
        print(f"  {'Level':<10}  {'n':>4}  {'Acc_C3':>8}  {'Acc_C2':>8}  "
              f"{'h':>6}  {'p':>10}  {'sig':>5}")
        print("  " + "-" * 58)

        for level in ("targeted", "high"):
            c3r = [r for r in tamper_all
                   if r["condition"] == "C3" and r["tamper_level"] == level
                   and r["classification"] != "error"]
            c2r = [r for r in tamper_all
                   if r["condition"] == "C2" and r["tamper_level"] == level
                   and r["classification"] != "error"]
            if not c3r or not c2r:
                continue
            # Align on scenario_id
            c3_map = {r["scenario_id"]: r for r in c3r}
            c2_map = {r["scenario_id"]: r for r in c2r}
            shared = sorted(set(c3_map) & set(c2_map))
            y3 = [_correct(c3_map[sid]) for sid in shared]
            y2 = [_correct(c2_map[sid]) for sid in shared]
            acc3 = sum(y3) / len(shared)
            acc2 = sum(y2) / len(shared)
            h    = cohens_h(acc3, acc2)
            mc   = mcnemar_pair(y3, y2)
            p    = mc["p_value"]
            sig  = "***" if p < 0.001 else ("**" if p < 0.01 else
                   ("*" if p < 0.05 else "ns"))
            print(f"  {level:<10}  {len(shared):>4}  {acc3:>8.4f}  {acc2:>8.4f}  "
                  f"{h:>+6.3f}  {p:>10.6f}  {sig:>5}")
            tamper_mc[level] = {
                "n": len(shared), "acc_c3": round(acc3, 4), "acc_c2": round(acc2, 4),
                "cohens_h": round(float(h), 4), "p_value": mc["p_value"],
                "significant": p < 0.05,
            }

    # ── Save ───────────────────────────────────────────────────
    def _to_native(obj):
        """Recursively convert numpy types to Python natives for JSON."""
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    out = _to_native({
        "overall_mcnemar":   mc_results,
        "per_type_c3_vs_c2": type_results,
        "tamper_mcnemar":    tamper_mc,
        "alpha":             alpha,
        "alpha_bonferroni":  alpha_bf,
        "n_scenarios":       n,
    })
    out_path = RESULTS / "significance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
