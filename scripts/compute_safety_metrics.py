"""
compute_safety_metrics.py

Safety-oriented decision metrics and paired uncertainty estimates for the
v14 battery task, addressing two reviewer requests:

  * "Urgent-event recall, under-triage rate, and cost-sensitive safety
    metrics should also be reported."
  * "Confidence intervals for paired performance differences should also
    be reported to better characterize the uncertainty of the
    decision-level improvement."

Definitions
-----------
Actions are ordered by severity:  safe (0) < replace (1) < urgent (2).

  under-triage    predicted severity < true severity.  The
                  safety-critical error: the system acts less
                  aggressively than the battery state requires.
  over-triage     predicted severity > true severity.  The cost is
                  operational (an unnecessary intervention), not safety.
  urgent recall   fraction of true-urgent cases assigned "urgent".
  actionable
    recall        fraction of true-actionable cases (replace or urgent)
                  not dismissed as "safe".

Cost-sensitive risk uses an asymmetric matrix in which a missed runaway
dominates.  Because any single matrix is arbitrary, the ranking is swept
over the missed-runaway cost across three orders of magnitude and the
ordering of conditions is reported as a function of that cost, rather
than at one hand-picked point.

Uncertainty
-----------
Accuracies carry Wilson score intervals.  Differences between conditions
are paired on scenario_id and estimated by a bootstrap over scenarios
(the resampling unit is the scenario, not the trial, because the same
scenario is evaluated under every condition).

Output: results/safety_metrics.json
"""

from __future__ import annotations
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RESULTS = Path("results")
OUT     = RESULTS / "safety_metrics.json"

SEED       = 42
N_BOOT     = 10000
ACTIONS    = ["safe", "replace", "urgent"]
SEVERITY   = {"safe": 0, "replace": 1, "urgent": 2}
ACTIONABLE = ("replace", "urgent")

# Source files -> condition label override (None keeps the record's own).
SOURCES: List[Tuple[str, Optional[Dict[str, str]]]] = [
    ("v14_poisoning_results.json",               None),
    ("v14_poisoning_c4_results.json",            None),
    ("v14_poisoning_c3_deployable_results.json", {"C3": "C3d"}),
    ("v14_verification_transfer_results.json",    None),
]

# Cost matrix: cost[true][pred].  Correct decisions cost nothing.
#   over-triage  : operational cost of an unnecessary intervention
#   under-triage : safety cost of an insufficient response
COST_OVER_1  = 1.0     # replace scheduled when safe would do
COST_OVER_2  = 5.0     # asset pulled offline unnecessarily
COST_UNDER_1 = 20.0    # replacement delayed
COST_MISS    = 100.0   # true urgent handled as safe: missed runaway


def cost_matrix(c_miss: float = COST_MISS) -> Dict[str, Dict[str, float]]:
    return {
        "safe":    {"safe": 0.0,          "replace": COST_OVER_1,
                    "urgent": COST_OVER_2},
        "replace": {"safe": COST_UNDER_1, "replace": 0.0,
                    "urgent": COST_OVER_1},
        "urgent":  {"safe": c_miss,       "replace": COST_UNDER_1,
                    "urgent": 0.0},
    }


# ── Interval estimates ────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_delta(
    a: Dict[str, int], b: Dict[str, int], n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Optional[Dict[str, float]]:
    """
    Bootstrap CI for mean(a) - mean(b) over the scenarios both conditions
    cover.  Resamples scenarios, preserving the pairing.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = [a[k] - b[k] for k in keys]
    n = len(diffs)
    point = sum(diffs) / n

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    # Fraction of bootstrap replicates on the other side of zero.
    if point >= 0:
        p_side = sum(1 for x in boots if x <= 0) / n_boot
    else:
        p_side = sum(1 for x in boots if x >= 0) / n_boot
    return {"delta": point, "ci_low": lo, "ci_high": hi,
            "n_pairs": n, "boot_p_one_sided": p_side}


# ── Metric computation ────────────────────────────────────────

def metrics_for(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the safety metric bundle for one group of records."""
    recs = [r for r in recs if r.get("classification") != "error"]
    n = len(recs)
    if n == 0:
        return {"n": 0}

    conf = {t: {p: 0 for p in ACTIONS} for t in ACTIONS}
    n_correct = under = over = 0
    unparsable = 0

    for r in recs:
        gt = r["ground_truth"]
        pd = r["decision"]
        if pd not in SEVERITY:
            unparsable += 1
            # An unusable output cannot act; count it as the most
            # conservative failure (no intervention taken).
            pd = "safe"
        if gt in conf:
            conf[gt][pd] += 1
        if pd == gt:
            n_correct += 1
        elif SEVERITY[pd] < SEVERITY[gt]:
            under += 1
        else:
            over += 1

    # Per-class recall / precision
    per_class = {}
    recalls = []
    for a in ACTIONS:
        tp = conf[a][a]
        support = sum(conf[a].values())
        pred_a = sum(conf[t][a] for t in ACTIONS)
        rec = tp / support if support else None
        prec = tp / pred_a if pred_a else None
        if rec is not None:
            recalls.append(rec)
        f1 = (2 * prec * rec / (prec + rec)
              if prec and rec and (prec + rec) > 0 else 0.0)
        per_class[a] = {"n": support, "recall": rec,
                        "precision": prec, "f1": f1}

    # Actionable subset
    act = [r for r in recs if r["ground_truth"] in ACTIONABLE]
    act_correct = sum(1 for r in act if r["decision"] == r["ground_truth"])
    act_not_dismissed = sum(
        1 for r in act
        if r["decision"] in SEVERITY and SEVERITY[r["decision"]] > 0)

    # Safe subset: false-intervention rate
    safe_recs = [r for r in recs if r["ground_truth"] == "safe"]
    false_int = sum(1 for r in safe_recs if r["decision"] != "safe")

    # Cost sweep
    costs = {}
    for c_miss in (10.0, 50.0, 100.0, 500.0, 1000.0):
        m = cost_matrix(c_miss)
        total = 0.0
        for t in ACTIONS:
            for p in ACTIONS:
                total += conf[t][p] * m[t][p]
        costs["c_miss_%g" % c_miss] = total / n

    out = {
        "n": n,
        "unparsable": unparsable,
        "accuracy": n_correct / n,
        "accuracy_ci": wilson(n_correct, n),
        "balanced_accuracy": (sum(recalls) / len(recalls)) if recalls else None,
        "macro_f1": sum(per_class[a]["f1"] for a in ACTIONS) / len(ACTIONS),
        "under_triage_rate": under / n,
        "over_triage_rate": over / n,
        "urgent_recall": per_class["urgent"]["recall"],
        "urgent_recall_ci": wilson(conf["urgent"]["urgent"],
                                   sum(conf["urgent"].values())),
        "actionable_n": len(act),
        "actionable_accuracy": (act_correct / len(act)) if act else None,
        "actionable_accuracy_ci": wilson(act_correct, len(act)) if act else None,
        "actionable_recall": (act_not_dismissed / len(act)) if act else None,
        "false_intervention_rate": (false_int / len(safe_recs)
                                    if safe_recs else None),
        "per_class": per_class,
        "confusion": conf,
        "expected_cost": costs,
    }
    return out


def correctness_index(recs: List[Dict[str, Any]],
                      actionable_only: bool) -> Dict[Tuple, Dict[str, int]]:
    """(condition, rate) -> {scenario_id: 0/1 correct}."""
    idx: Dict[Tuple, Dict[str, int]] = defaultdict(dict)
    for r in recs:
        if r.get("classification") == "error":
            continue
        if actionable_only and r["ground_truth"] not in ACTIONABLE:
            continue
        idx[(r["condition"], r["rate"])][r["scenario_id"]] = (
            1 if r["decision"] == r["ground_truth"] else 0)
    return idx


def main() -> None:
    recs: List[Dict[str, Any]] = []
    loaded = []
    for fname, relabel in SOURCES:
        p = RESULTS / fname
        if not p.exists():
            print("  skip (missing): %s" % fname)
            continue
        data = json.loads(p.read_text())
        for r in data:
            if relabel and r["condition"] in relabel:
                r["condition"] = relabel[r["condition"]]
        recs.extend(data)
        loaded.append("%s (%d)" % (fname, len(data)))
    print("Loaded: " + ", ".join(loaded))

    conditions = sorted({r["condition"] for r in recs})
    rates      = sorted({r["rate"] for r in recs})

    out: Dict[str, Any] = {
        "sources": loaded,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "cost_matrix_note": (
            "cost[true][pred]; over-triage 1/5, under-triage 20, "
            "missed runaway swept over 10-1000"),
        "by_condition_rate": {},
        "poisoned_only": {},
        "paired_deltas": {},
    }

    # -- per condition x rate --------------------------------
    print("\n%-5s %-6s %6s %8s %8s %8s %8s %9s"
          % ("cond", "rate", "n", "acc", "act_acc", "urg_rec",
             "under", "cost100"))
    for c in conditions:
        for rate in rates:
            grp = [r for r in recs
                   if r["condition"] == c and r["rate"] == rate]
            if not grp:
                continue
            m = metrics_for(grp)
            out["by_condition_rate"]["%s@%.2f" % (c, rate)] = m
            print("%-5s %-6.2f %6d %8.3f %8s %8s %8.3f %9.2f"
                  % (c, rate, m["n"], m["accuracy"],
                     ("%.3f" % m["actionable_accuracy"]
                      if m["actionable_accuracy"] is not None else "NA"),
                     ("%.3f" % m["urgent_recall"]
                      if m["urgent_recall"] is not None else "NA"),
                     m["under_triage_rate"],
                     m["expected_cost"]["c_miss_100"]))

            pois = [r for r in grp if r.get("poisoned")]
            if pois:
                out["poisoned_only"]["%s@%.2f" % (c, rate)] = metrics_for(pois)

    # -- paired bootstrap deltas -----------------------------
    for subset_name, actionable_only in (("overall", False),
                                         ("actionable", True)):
        idx = correctness_index(recs, actionable_only)
        block: Dict[str, Any] = {}
        for rate in rates:
            for i, ca in enumerate(conditions):
                for cb in conditions[i + 1:]:
                    a = idx.get((ca, rate))
                    b = idx.get((cb, rate))
                    if not a or not b:
                        continue
                    d = paired_bootstrap_delta(a, b)
                    if d:
                        block["%s-%s@%.2f" % (ca, cb, rate)] = d
        out["paired_deltas"][subset_name] = block

    # Headline pairs at the strongest corruption rate.
    print("\nPaired bootstrap, actionable subset @ rate 0.50 "
          "(delta, 95%% CI):")
    for k, v in sorted(out["paired_deltas"]["actionable"].items()):
        if k.endswith("@0.50"):
            print("  %-14s %+.3f  [%+.3f, %+.3f]  n=%d"
                  % (k[:-5], v["delta"], v["ci_low"], v["ci_high"],
                     v["n_pairs"]))

    OUT.write_text(json.dumps(out, indent=2))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
