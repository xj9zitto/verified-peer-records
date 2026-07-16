"""
run_ablation.py
Ablation studies — k-retrieval and prompt-variation (metrics 8 & 9).

Metric 8 — k-ablation:
  Run C3 with k ∈ {3, 5, 8} (total retrieved records).
  k=8 is the baseline (already in raw_results.json); k=3 and k=5 are new.
  k→(k_asset, k_routing): 3→(2,1), 5→(3,2), 8→(5,3).

Metric 9 — Prompt variation stability:
  Run C3 with three phrasings of the decision instruction (v1/v2/v3).
  v1 is the baseline (already in raw_results.json); v2 and v3 are new.

Usage:
    python scripts/run_ablation.py                  # both studies, all 120 scenarios
    python scripts/run_ablation.py --study k        # k-ablation only
    python scripts/run_ablation.py --study prompt   # prompt variation only
    python scripts/run_ablation.py --limit 20       # smoke test
    python scripts/run_ablation.py --resume         # skip already-done runs
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from tqdm import tqdm
from loguru import logger

from src.logger_setup import setup_logger
from src.layer3.llm_agent import LLMAgent

setup_logger("ablation")

RESULTS_DIR = Path("results")

# k-ablation: total k → (k_asset, k_routing), keeping ~5:3 asset:routing ratio
K_CONFIGS: Dict[int, Tuple[int, int]] = {
    3:  (2,  1),
    5:  (3,  2),
    8:  (5,  3),   # baseline — already in raw_results.json
    12: (8,  4),
    16: (10, 6),
}

PROMPT_VARIANTS = ["v1", "v2", "v3"]   # v1 is baseline


# ── Ground-truth normaliser (mirrors metrics.py) ──────────────

_GT_MAP = {
    "r1": "reroute", "r2": "reroute", "r3": "reroute", "r4": "reroute",
    "reroute": "reroute",
    "no_safe_route_available": "no_safe_route",
    "no_safe_route":           "no_safe_route",
    "reorder":                 "reorder",
    "reorder_degradation_flag":"reorder",
    "replenish":               "reorder",
    "k1": "carrier", "k2": "carrier", "k3": "carrier",
    "use k1": "carrier", "use k2": "carrier", "use k3": "carrier",
    "carrier": "carrier",
    "hold":    "hold",
    "uncertain":"uncertain",
    "error":   "error",
}


def _norm(label: str) -> str:
    return _GT_MAP.get(str(label).lower().strip(), str(label).lower().strip())


# ── Checkpoint helpers ─────────────────────────────────────────

def _load_file(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_file(path: Path, records: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def _done_set(records: List[Dict], key_field: str) -> set:
    """Return the set of already-completed scenario IDs."""
    return {r[key_field] for r in records}


# ── Single-run helper ──────────────────────────────────────────

def _run_scenario(
    agent:          LLMAgent,
    scenario:       Dict[str, Any],
    k_asset:        int,
    k_routing:      int,
    prompt_variant: str,
) -> Dict[str, Any]:
    result = agent.decide(
        scenario,
        condition="C3",
        k_asset=k_asset,
        k_routing=k_routing,
        prompt_variant=prompt_variant,
    )
    return {
        "scenario_id":         scenario["scenario_id"],
        "scenario_type":       scenario.get("scenario_type", ""),
        "ground_truth":        scenario.get("ground_truth", ""),
        "is_multi_constraint": scenario.get("is_multi_constraint", False),
        **result,
    }


# ── Study 1: k-ablation ────────────────────────────────────────

def run_k_ablation(
    scenarios: List[Dict],
    agent:     LLMAgent,
    resume:    bool = True,
) -> Dict[int, List[Dict]]:
    """
    Run C3 with k ∈ {3, 5} (k=8 baseline already exists).
    Returns {k_total: [result_records]}.
    """
    results: Dict[int, List[Dict]] = {}

    for k_total, (k_asset, k_routing) in K_CONFIGS.items():
        if k_total == 8:
            # Load baseline results from main experiment for k=8
            baseline_path = RESULTS_DIR / "raw_results.json"
            if baseline_path.exists():
                with open(baseline_path) as f:
                    all_results = json.load(f)
                c3_baseline = [
                    r for r in all_results
                    if r["condition"] == "C3"
                    and r["classification"] != "error"
                ]
                if c3_baseline:
                    results[8] = c3_baseline
                    print(f"  k=8: loaded {len(c3_baseline)} baseline C3 records")
                    continue

        out_path = RESULTS_DIR / f"ablation_k{k_total}.json"
        existing = _load_file(out_path) if resume else []
        done     = _done_set(existing, "scenario_id")

        todo = [s for s in scenarios if s["scenario_id"] not in done]
        print(f"\n  k={k_total} (k_asset={k_asset}, k_routing={k_routing}): "
              f"{len(todo)} runs remaining")

        bar = tqdm(todo, desc=f"k={k_total}", unit="run", dynamic_ncols=True)
        for scenario in bar:
            try:
                rec = _run_scenario(agent, scenario, k_asset, k_routing, "v1")
            except Exception as exc:
                logger.error(f"k={k_total} {scenario['scenario_id'][:8]}: {exc}")
                rec = {
                    "scenario_id":   scenario["scenario_id"],
                    "scenario_type": scenario.get("scenario_type", ""),
                    "ground_truth":  scenario.get("ground_truth", ""),
                    "is_multi_constraint": scenario.get("is_multi_constraint", False),
                    "decision": "error", "classification": "error",
                    "trace_rate": 0.0, "latency_ms": 0.0,
                    "latency_retrieval_ms": 0.0, "latency_inference_ms": 0.0,
                    "model_used": "error", "tamper_alerts": 0,
                    "condition": "C3", "llm_output": str(exc),
                    "violations": [], "k_asset": k_asset, "k_routing": k_routing,
                    "prompt_variant": "v1",
                }

            existing.append(rec)
            _save_file(out_path, existing)

            correct = _norm(rec["ground_truth"]) == _norm(rec["decision"])
            bar.set_postfix(acc="✓" if correct else "✗", tau=f"{rec['trace_rate']:.2f}")

        results[k_total] = [r for r in existing if r["classification"] != "error"]
        print(f"  k={k_total}: saved {len(existing)} records → {out_path}")

    return results


# ── Study 2: Prompt variation ──────────────────────────────────

def run_prompt_variation(
    scenarios: List[Dict],
    agent:     LLMAgent,
    resume:    bool = True,
) -> Dict[str, List[Dict]]:
    """
    Run C3 with prompt variants v2 and v3 (v1 is the baseline).
    Returns {variant: [result_records]}.
    """
    results: Dict[str, List[Dict]] = {}

    # Load v1 baseline from main experiment
    baseline_path = RESULTS_DIR / "raw_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            all_results = json.load(f)
        v1_records = [
            r for r in all_results
            if r["condition"] == "C3"
            and r["classification"] != "error"
        ]
        if v1_records:
            results["v1"] = v1_records
            print(f"  v1: loaded {len(v1_records)} baseline C3 records")

    for variant in ["v2", "v3"]:
        out_path = RESULTS_DIR / f"ablation_prompt_{variant}.json"
        existing = _load_file(out_path) if resume else []
        done     = _done_set(existing, "scenario_id")

        todo = [s for s in scenarios if s["scenario_id"] not in done]
        print(f"\n  {variant}: {len(todo)} runs remaining")

        bar = tqdm(todo, desc=f"prompt {variant}", unit="run", dynamic_ncols=True)
        for scenario in bar:
            try:
                rec = _run_scenario(agent, scenario, 5, 3, variant)
            except Exception as exc:
                logger.error(f"{variant} {scenario['scenario_id'][:8]}: {exc}")
                rec = {
                    "scenario_id":   scenario["scenario_id"],
                    "scenario_type": scenario.get("scenario_type", ""),
                    "ground_truth":  scenario.get("ground_truth", ""),
                    "is_multi_constraint": scenario.get("is_multi_constraint", False),
                    "decision": "error", "classification": "error",
                    "trace_rate": 0.0, "latency_ms": 0.0,
                    "latency_retrieval_ms": 0.0, "latency_inference_ms": 0.0,
                    "model_used": "error", "tamper_alerts": 0,
                    "condition": "C3", "llm_output": str(exc),
                    "violations": [], "k_asset": 5, "k_routing": 3,
                    "prompt_variant": variant,
                }

            existing.append(rec)
            _save_file(out_path, existing)

            correct = _norm(rec["ground_truth"]) == _norm(rec["decision"])
            bar.set_postfix(pv=variant, acc="✓" if correct else "✗",
                            tau=f"{rec['trace_rate']:.2f}")

        results[variant] = [r for r in existing if r["classification"] != "error"]
        print(f"  {variant}: saved {len(existing)} records → {out_path}")

    return results


# ── Result printers ────────────────────────────────────────────

def print_k_ablation_table(results: Dict[int, List[Dict]]) -> None:
    print("\n" + "=" * 58)
    print("  METRIC 8 — k-Ablation (C3, k total retrieved records)")
    print("=" * 58)
    print(f"  {'k':>4}  {'n':>5}  {'Accuracy':>9}  {'Trace τ':>8}  {'Latency(ms)':>11}")
    print("  " + "-" * 52)

    for k_total in sorted(results):
        recs = results[k_total]
        if not recs:
            continue
        correct = sum(
            1 for r in recs
            if _norm(r["ground_truth"]) == _norm(r["decision"])
        )
        acc = correct / len(recs)
        tau = np.mean([r["trace_rate"] for r in recs])
        lat = np.mean([r["latency_ms"] for r in recs])
        mark = " ← baseline" if k_total == 8 else ""
        print(f"  {k_total:>4}  {len(recs):>5}  {acc:>9.3f}  {tau:>8.3f}  {lat:>11.0f}{mark}")

    print("=" * 58)

    # Save to results/ablation_k_summary.json
    summary = {}
    for k_total, recs in results.items():
        if not recs:
            continue
        correct = sum(1 for r in recs
                      if _norm(r["ground_truth"]) == _norm(r["decision"]))
        summary[str(k_total)] = {
            "n":        len(recs),
            "accuracy": round(correct / len(recs), 4),
            "tau":      round(float(np.mean([r["trace_rate"] for r in recs])), 4),
            "latency_ms": round(float(np.mean([r["latency_ms"] for r in recs])), 1),
        }
    out = RESULTS_DIR / "ablation_k_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {out}")


def print_prompt_variation_table(results: Dict[str, List[Dict]]) -> None:
    print("\n" + "=" * 62)
    print("  METRIC 9 — Prompt Variation Stability (C3, k=8)")
    print("=" * 62)
    print(f"  {'Variant':<10}  {'n':>5}  {'Accuracy':>9}  {'Trace τ':>8}  {'Latency(ms)':>11}")
    print("  " + "-" * 56)

    DESCRIPTIONS = {
        "v1": "structured (baseline)",
        "v2": "step-by-step",
        "v3": "concise natural",
    }

    for variant in ["v1", "v2", "v3"]:
        recs = results.get(variant, [])
        if not recs:
            continue
        correct = sum(
            1 for r in recs
            if _norm(r["ground_truth"]) == _norm(r["decision"])
        )
        acc = correct / len(recs)
        tau = np.mean([r["trace_rate"] for r in recs])
        lat = np.mean([r["latency_ms"] for r in recs])
        desc = DESCRIPTIONS.get(variant, "")
        print(f"  {variant:<10}  {len(recs):>5}  {acc:>9.3f}  {tau:>8.3f}  "
              f"{lat:>11.0f}  {desc}")

    print("=" * 62)

    # Compute stability: std of accuracy across variants
    accs = []
    for variant, recs in results.items():
        if not recs:
            continue
        correct = sum(1 for r in recs
                      if _norm(r["ground_truth"]) == _norm(r["decision"]))
        accs.append(correct / len(recs))
    if len(accs) > 1:
        print(f"\n  Accuracy std across variants: {np.std(accs):.4f}  "
              f"(lower = more stable)")

    # Save summary
    summary = {}
    for variant, recs in results.items():
        if not recs:
            continue
        correct = sum(1 for r in recs
                      if _norm(r["ground_truth"]) == _norm(r["decision"]))
        summary[variant] = {
            "n":          len(recs),
            "accuracy":   round(correct / len(recs), 4),
            "tau":        round(float(np.mean([r["trace_rate"] for r in recs])), 4),
            "latency_ms": round(float(np.mean([r["latency_ms"] for r in recs])), 1),
        }
    out = RESULTS_DIR / "ablation_prompt_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {out}")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation studies (metrics 8 & 9)")
    parser.add_argument("--study",  choices=["k", "prompt", "both"], default="both",
                        help="Which study to run (default: both)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Run only first N scenarios (smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed runs")
    args = parser.parse_args()

    with open("data/synthetic/scenarios.json") as f:
        scenarios = json.load(f)
    if args.limit:
        scenarios = scenarios[: args.limit]

    print(f"\nAblation study — {len(scenarios)} scenarios, study={args.study}")
    print("Loading LLM agent...")
    agent = LLMAgent()

    if args.study in ("k", "both"):
        print("\n=== Study 1: k-Ablation ===")
        k_results = run_k_ablation(scenarios, agent, resume=args.resume)
        print_k_ablation_table(k_results)

    if args.study in ("prompt", "both"):
        print("\n=== Study 2: Prompt Variation ===")
        pv_results = run_prompt_variation(scenarios, agent, resume=args.resume)
        print_prompt_variation_table(pv_results)


if __name__ == "__main__":
    main()
