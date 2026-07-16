"""
metrics.py
Evaluation — all 10 evaluation metrics with bootstrapped 95% CIs.

Input: raw_results.json produced by conditions.py
Output: metrics dict, printed summary, and results/metrics.json
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import BOOTSTRAP_SAMPLES, CONDITIONS

RESULTS_DIR  = Path("results")
METRICS_FILE = RESULTS_DIR / "metrics.json"

# Ground-truth label normaliser — maps both ground-truth IDs and LLM action
# words to a canonical set: reorder | reroute | carrier | no_safe_route | hold
_GT_MAP = {
    # ── Route / reroute decisions ──────────────────────────────
    "r1": "reroute", "r2": "reroute", "r3": "reroute", "r4": "reroute",
    "reroute":       "reroute",
    "reroute to r1": "reroute", "reroute to r2": "reroute",
    "reroute to r3": "reroute", "reroute to r4": "reroute",
    "use route r1":  "reroute", "use route r2":  "reroute",
    "use route r3":  "reroute", "use route r4":  "reroute",
    "switch route":  "reroute", "change route":  "reroute",
    # ── No-safe-route escalation ───────────────────────────────
    "no_safe_route_available": "no_safe_route",
    "no_safe_route":           "no_safe_route",
    # ── Reorder decisions ──────────────────────────────────────
    "reorder":                 "reorder",
    "reorder_degradation_flag":"reorder",   # ground-truth label variant
    "replenish":               "reorder",
    # ── Carrier decisions ──────────────────────────────────────
    "k1": "carrier", "k2": "carrier", "k3": "carrier",
    "use k1": "carrier", "use k2": "carrier", "use k3": "carrier",
    "carrier":  "carrier",
    # ── Hold ──────────────────────────────────────────────────
    "hold": "hold",
    # ── Error / uncertain ─────────────────────────────────────
    "uncertain": "uncertain",
    "error":     "error",
}


def _norm(label: str) -> str:
    return _GT_MAP.get(label.lower().strip(), label.lower().strip())


# ── Bootstrap CI ───────────────────────────────────────────────

def _bootstrap_ci(
    values: List[float],
    n:      int = BOOTSTRAP_SAMPLES,
    alpha:  float = 0.05,
) -> Tuple[float, float]:
    """Return (lower, upper) 95% bootstrap CI for the mean."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
             for _ in range(n)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (round(lo, 4), round(hi, 4))


# ── Per-condition helpers ──────────────────────────────────────

def _by_condition(results: List[Dict]) -> Dict[str, List[Dict]]:
    d = defaultdict(list)
    for r in results:
        d[r["condition"]].append(r)
    return d


# ── Individual metric functions ────────────────────────────────

def accuracy(records: List[Dict]) -> Dict[str, Any]:
    """Metric 1 — decision accuracy vs physics-labelled ground truth."""
    correct = [
        1.0 if _norm(r["decision"]) == _norm(r["ground_truth"]) else 0.0
        for r in records if r["classification"] != "error"
    ]
    mean = float(np.mean(correct)) if correct else 0.0
    ci   = _bootstrap_ci(correct)
    return {"mean": round(mean, 4), "ci_95": ci, "n": len(correct)}


def hallucination_rate(records: List[Dict]) -> Dict[str, Any]:
    """
    Metric 2 — hallucination rate.
    A decision is hallucinated when the model commits to an action
    (not uncertain) but trace_rate == 0 (no claims traceable to records).
    """
    flags = []
    for r in records:
        if r["classification"] == "error":
            continue
        committed = r["classification"] != "uncertain"
        no_trace  = r["trace_rate"] == 0.0
        # C0/C1 have no retrieval; only flag C2/C3
        if r["condition"] in ("C2", "C3"):
            flags.append(1.0 if (committed and no_trace) else 0.0)
    mean = float(np.mean(flags)) if flags else 0.0
    ci   = _bootstrap_ci(flags)
    return {"mean": round(mean, 4), "ci_95": ci, "n": len(flags)}


def uncertain_rate(records: List[Dict]) -> Dict[str, Any]:
    """Metric 3 — rate at which the model returns 'uncertain'."""
    flags = [
        1.0 if r["classification"] == "uncertain" else 0.0
        for r in records if r["classification"] != "error"
    ]
    mean = float(np.mean(flags)) if flags else 0.0
    ci   = _bootstrap_ci(flags)
    return {"mean": round(mean, 4), "ci_95": ci, "n": len(flags)}


def mean_trace_rate(records: List[Dict]) -> Dict[str, Any]:
    """Metric 4 — mean τ (fraction of claims traced to chain_tx_hash)."""
    vals = [
        r["trace_rate"]
        for r in records
        if r["condition"] in ("C2", "C3") and r["classification"] != "error"
    ]
    mean = float(np.mean(vals)) if vals else 0.0
    ci   = _bootstrap_ci(vals)
    return {"mean": round(mean, 4), "ci_95": ci, "n": len(vals)}


def robustness_delta(
    c2_records: List[Dict],
    c3_records: List[Dict],
) -> Dict[str, Any]:
    """
    Metric 5 — accuracy delta C3 − C2 (robustness under corruption).
    Positive value means C3 is more accurate (expected).
    """
    acc_c2 = accuracy(c2_records)["mean"]
    acc_c3 = accuracy(c3_records)["mean"]
    delta  = round(acc_c3 - acc_c2, 4)
    return {
        "acc_c2": acc_c2,
        "acc_c3": acc_c3,
        "delta_c3_minus_c2": delta,
    }


def latency_breakdown(records: List[Dict]) -> Dict[str, Any]:
    """
    Metric 6 — mean latency breakdown per condition.
    Returns retrieval_ms, inference_ms, total_ms.
    """
    result = {}
    for cond, recs in _by_condition(records).items():
        valid = [r for r in recs if r["classification"] != "error"]
        if not valid:
            continue
        result[cond] = {
            "retrieval_ms": round(
                float(np.mean([r["latency_retrieval_ms"] for r in valid])), 1
            ),
            "inference_ms": round(
                float(np.mean([r["latency_inference_ms"] for r in valid])), 1
            ),
            "total_ms": round(
                float(np.mean([r["latency_ms"] for r in valid])), 1
            ),
        }
    return result


def cohens_kappa(
    a_records: List[Dict],
    b_records: List[Dict],
    label_a:   str = "C2",
    label_b:   str = "C3",
) -> Dict[str, Any]:
    """
    Metric 7 — Cohen's κ between two conditions on the same scenarios.
    Measures inter-rater agreement on the decision labels.
    """
    # Align on scenario_id
    a_map = {r["scenario_id"]: _norm(r["decision"]) for r in a_records}
    b_map = {r["scenario_id"]: _norm(r["decision"]) for r in b_records}
    shared = sorted(set(a_map) & set(b_map))

    if not shared:
        return {"kappa": None, "n": 0}

    labels = sorted({a_map[s] for s in shared} | {b_map[s] for s in shared})
    label_idx = {l: i for i, l in enumerate(labels)}
    n_labels  = len(labels)
    n         = len(shared)

    # Confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=float)
    for s in shared:
        i = label_idx[a_map[s]]
        j = label_idx[b_map[s]]
        cm[i, j] += 1

    p_o = np.trace(cm) / n
    p_e = float(np.sum((cm.sum(axis=1) / n) * (cm.sum(axis=0) / n)))
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0

    return {
        "kappa": round(float(kappa), 4),
        f"acc_{label_a}": round(float(np.trace(cm) / n), 4),
        "n": n,
        "labels": labels,
        "pair": f"{label_a}_vs_{label_b}",
    }


def factual_conflict_rate(records: List[Dict]) -> Dict[str, Any]:
    """
    Metric 10 — rate of decisions that violate physics thresholds.

    Counts ONLY threshold violations (SOC/temp/degradation constraint
    breached). Untraceable claims in no-retrieval conditions (C0/C1)
    are excluded because they are structurally expected, not model errors.
    """
    flags = []
    for r in records:
        if r["classification"] == "error":
            continue
        violations = r.get("violations", [])
        # Only count actual physics threshold violations, not traceability misses
        has_physics_violation = any(
            v != "untraceable_claim" and v != "no_verifiable_numerical_claim"
            for v in violations
        )
        flags.append(1.0 if has_physics_violation else 0.0)
    mean = float(np.mean(flags)) if flags else 0.0
    ci   = _bootstrap_ci(flags)
    return {"mean": round(mean, 4), "ci_95": ci, "n": len(flags)}


def untraceable_rate(records: List[Dict]) -> Dict[str, Any]:
    """
    Supplementary — rate of decisions with untraceable numerical claims.
    High in C1 (by design), low in C3 (desired).
    """
    flags = [
        1.0 if "untraceable_claim" in r.get("violations", []) else 0.0
        for r in records if r["classification"] != "error"
    ]
    mean = float(np.mean(flags)) if flags else 0.0
    ci   = _bootstrap_ci(flags)
    return {"mean": round(mean, 4), "ci_95": ci, "n": len(flags)}


# ── Main compute function ──────────────────────────────────────

def compute(
    results:  List[Dict[str, Any]],
    save:     bool = True,
    verbose:  bool = True,
) -> Dict[str, Any]:
    """
    Compute all 10 metrics from raw experiment results.

    Parameters
    ----------
    results : list of result records (from conditions.run())
    save    : write metrics.json to results/
    verbose : print formatted summary table

    Returns
    -------
    Nested dict: metrics[condition][metric_name] = value
    """
    by_cond = _by_condition(results)

    metrics: Dict[str, Any] = {}

    for cond in CONDITIONS:
        recs = by_cond.get(cond, [])
        if not recs:
            continue
        metrics[cond] = {
            "n":                   len(recs),
            "accuracy":            accuracy(recs),
            "hallucination_rate":  hallucination_rate(recs),
            "uncertain_rate":      uncertain_rate(recs),
            "trace_rate":          mean_trace_rate(recs),
            "factual_conflict":    factual_conflict_rate(recs),
            "untraceable_rate":    untraceable_rate(recs),
            "latency":             latency_breakdown(recs),
        }

    # Cross-condition metrics
    c2 = by_cond.get("C2", [])
    c3 = by_cond.get("C3", [])
    c0 = by_cond.get("C0", [])
    c1 = by_cond.get("C1", [])

    metrics["robustness"] = robustness_delta(c2, c3) if c2 and c3 else {}

    metrics["cohens_kappa"] = {}
    for (la, ra), (lb, rb) in [
        (("C0", c0), ("C3", c3)),
        (("C1", c1), ("C3", c3)),
        (("C2", c2), ("C3", c3)),
    ]:
        if ra and rb:
            metrics["cohens_kappa"][f"{la}_vs_{lb}"] = cohens_kappa(
                ra, rb, la, lb
            )

    # Per scenario-type breakdown
    metrics["by_type"] = {}
    for stype in ("reorder", "reroute", "path", "carrier"):
        metrics["by_type"][stype] = {}
        for cond in CONDITIONS:
            recs = [r for r in by_cond.get(cond, [])
                    if r["scenario_type"] == stype]
            if recs:
                metrics["by_type"][stype][cond] = {
                    "accuracy": accuracy(recs)["mean"],
                    "n": len(recs),
                }

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print(f"\nMetrics saved -> {METRICS_FILE}")

    if verbose:
        _print_summary(metrics)

    return metrics


# ── Pretty-print summary ───────────────────────────────────────

def _print_summary(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 62)
    print("  EXPERIMENT RESULTS")
    print("=" * 62)
    print(f"{'Metric':<28}  {'C0':>7}  {'C1':>7}  {'C2':>7}  {'C3':>7}")
    print("-" * 62)

    def _v(cond, key, subkey="mean"):
        try:
            val = metrics[cond][key]
            return f"{val[subkey]:.3f}" if subkey in val else f"{val:.3f}"
        except (KeyError, TypeError):
            return "  —  "

    rows = [
        ("Accuracy",             "accuracy"),
        ("Hallucination rate",   "hallucination_rate"),
        ("Uncertain rate",       "uncertain_rate"),
        ("Trace rate τ",         "trace_rate"),
        ("Factual conflict",     "factual_conflict"),
        ("Untraceable rate",     "untraceable_rate"),
    ]
    for label, key in rows:
        vals = "  ".join(f"{_v(c, key):>7}" for c in CONDITIONS)
        print(f"  {label:<26}{vals}")

    print("-" * 62)
    # Latency (total_ms)
    lat_vals = []
    for c in CONDITIONS:
        try:
            v = metrics[c]["latency"].get(c, {}).get("total_ms", None)
            lat_vals.append(f"{v:>7.0f}" if v else "  —  ")
        except (KeyError, TypeError):
            lat_vals.append("  —  ")
    print(f"  {'Latency total (ms)':<26}" + "  ".join(lat_vals))

    print("-" * 62)
    rob = metrics.get("robustness", {})
    if rob:
        print(f"\n  Robustness (C3−C2 accuracy delta): "
              f"{rob.get('delta_c3_minus_c2', '—'):+.3f}")

    kap = metrics.get("cohens_kappa", {})
    for pair, v in kap.items():
        k = v.get("kappa")
        if k is not None:
            print(f"  Cohen's κ {pair}: {k:.3f}  (n={v['n']})")

    print("=" * 62)
