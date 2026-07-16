"""
conditions.py
Evaluation — experiment runner for all four conditions (C0–C3).

Runs 120 scenarios × {C0, C1, C2, C3} and saves one result record
per (scenario, condition) pair to results/raw_results.json.

Checkpoint design: results are appended and the file is re-written
after every scenario so the run can be resumed if interrupted.
C0 uses the physics rule engine (no LLM). C1/C2/C3 use LLMAgent.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.logger_setup import setup_logger
from src.config import (
    SOC_REORDER_THRESHOLD, DEGRADATION_REPLACE_FLAG,
)
from src.layer3.physics_checker import compute_t_safe, compute_t_arr
from src.layer3.llm_agent import LLMAgent

setup_logger("experiment")

RESULTS_DIR  = Path("results")
RAW_FILE     = RESULTS_DIR / "raw_results.json"
CONDITIONS   = ["C0", "C1", "C2", "C3"]


# ── C0 Rule Engine ─────────────────────────────────────────────

def _c0_decide(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure physics rule engine (C0 baseline, no LLM).
    Applies thresholds in priority order: SOC → temperature → degradation.
    Returns the same result-dict schema as LLMAgent.decide().
    """
    t0  = time.perf_counter()
    soc = scenario.get("soc_pct", 100.0)
    deg = scenario.get("degradation_pct", 100.0)
    t_safe = compute_t_safe(deg)

    # Compute max projected arrival temperature across routing records
    t_arr_max = 0.0
    for rec in scenario.get("routing_records", []):
        try:
            t_arr = compute_t_arr(
                float(rec["amb_temp_max_c"]),
                float(rec["carrier_variance_c"]),
                deg,
            )
            t_arr_max = max(t_arr_max, t_arr)
        except (KeyError, TypeError, ValueError):
            pass

    # Priority: SOC → temperature → degradation → hold
    if soc < SOC_REORDER_THRESHOLD:
        decision = "reorder"
    elif t_arr_max > t_safe:
        decision = "reroute"
    elif deg < DEGRADATION_REPLACE_FLAG:
        decision = "carrier"   # flag for temperature-controlled carrier
    else:
        decision = "hold"

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "decision":             decision,
        "classification":       "valid",    # rule engine is always valid
        "trace_rate":           0.0,        # no chain tracing in C0
        "latency_ms":           round(latency_ms, 1),
        "latency_retrieval_ms": 0.0,
        "latency_inference_ms": 0.0,
        "model_used":           "rule_engine",
        "tamper_alerts":        0,
        "condition":            "C0",
        "llm_output":           "",
        "violations":           [],
    }


# ── Done-set helpers ───────────────────────────────────────────

def _load_done(path: Path) -> set:
    """Return set of (scenario_id, condition) tuples already in results."""
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    return {(r["scenario_id"], r["condition"]) for r in data}


def _append_result(path: Path, record: Dict[str, Any]) -> None:
    """Append one result record to the JSON file (checkpoint write)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing.append(record)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# ── Main runner ────────────────────────────────────────────────

def run(
    scenarios:  List[Dict[str, Any]],
    conditions: List[str] = CONDITIONS,
    agent:      Optional[LLMAgent] = None,
    output:     Path = RAW_FILE,
    resume:     bool = True,
) -> List[Dict[str, Any]]:
    """
    Run the full experiment: len(scenarios) × len(conditions) decisions.

    Parameters
    ----------
    scenarios  : list of scenario dicts from scenarios.json
    conditions : subset of ["C0","C1","C2","C3"] to run
    agent      : LLMAgent instance (created once, reused across all runs)
    output     : path to checkpoint / output JSON file
    resume     : if True, skip (scenario, condition) pairs already in output

    Returns
    -------
    List of result records (all completed so far, including prior runs).
    """
    if agent is None:
        agent = LLMAgent()

    done = _load_done(output) if resume else set()
    total = len(scenarios) * len(conditions)
    skipped = len(done)
    logger.info(
        f"Experiment start — {len(scenarios)} scenarios × "
        f"{len(conditions)} conditions = {total} runs "
        f"({skipped} already done, resuming={resume})"
    )

    bar = tqdm(
        total=total - skipped,
        desc="experiment",
        unit="run",
        dynamic_ncols=True,
    )

    for scenario in scenarios:
        sid = scenario["scenario_id"]

        for condition in conditions:
            if (sid, condition) in done:
                continue

            try:
                if condition == "C0":
                    result = _c0_decide(scenario)
                else:
                    result = agent.decide(scenario, condition=condition)

                record = {
                    "scenario_id":        sid,
                    "scenario_type":      scenario.get("scenario_type", ""),
                    "ground_truth":       scenario.get("ground_truth", ""),
                    "is_multi_constraint":scenario.get("is_multi_constraint", False),
                    **result,
                }

            except Exception as exc:
                logger.error(
                    f"FAILED {sid[:8]}... [{condition}]: {exc}"
                )
                record = {
                    "scenario_id":        sid,
                    "scenario_type":      scenario.get("scenario_type", ""),
                    "ground_truth":       scenario.get("ground_truth", ""),
                    "is_multi_constraint":scenario.get("is_multi_constraint", False),
                    "decision":           "error",
                    "classification":     "error",
                    "trace_rate":         0.0,
                    "latency_ms":         0.0,
                    "latency_retrieval_ms": 0.0,
                    "latency_inference_ms": 0.0,
                    "model_used":         "error",
                    "tamper_alerts":      0,
                    "condition":          condition,
                    "llm_output":         str(exc),
                    "violations":         [],
                }

            _append_result(output, record)
            done.add((sid, condition))
            bar.update(1)
            bar.set_postfix(
                cond=condition,
                cls=record["classification"][:3],
                tau=f"{record['trace_rate']:.2f}",
            )

    bar.close()

    with open(output) as f:
        all_results = json.load(f)

    logger.info(
        f"Experiment complete — {len(all_results)} records saved to {output}"
    )
    return all_results
