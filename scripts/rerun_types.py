"""
rerun_types.py
Re-run specified scenario types for specified conditions.

Usage:
    python scripts/rerun_types.py --types reorder carrier path --conditions C1 C2 C3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from src.logger_setup import setup_logger
from src.evaluation.conditions import _append_result, RAW_FILE
from src.layer3.llm_agent import LLMAgent

setup_logger("rerun_types")

RESULTS_DIR = Path("results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="+", default=["reorder", "carrier", "path"])
    parser.add_argument("--conditions", nargs="+", default=["C1", "C2", "C3"])
    args = parser.parse_args()

    stypes     = args.types
    conditions = args.conditions

    with open("data/synthetic/scenarios.json") as f:
        all_scenarios = json.load(f)

    target_scenarios = [s for s in all_scenarios if s["scenario_type"] in stypes]
    print(f"Re-running {len(target_scenarios)} scenarios "
          f"(types={stypes}) × {conditions}\n")

    raw_path = RAW_FILE
    if raw_path.exists():
        with open(raw_path) as f:
            existing = json.load(f)

        to_remove = {
            (r["scenario_id"], r["condition"])
            for r in existing
            if r["scenario_type"] in stypes and r["condition"] in conditions
        }
        kept = [
            r for r in existing
            if (r["scenario_id"], r["condition"]) not in to_remove
        ]
        print(f"Removing {len(to_remove)} old records from raw_results.json")
        with open(raw_path, "w") as f:
            json.dump(kept, f, indent=2)

    agent = LLMAgent()
    total = len(target_scenarios) * len(conditions)
    done  = 0

    for scenario in target_scenarios:
        sid   = scenario["scenario_id"]
        stype = scenario["scenario_type"]
        for cond in conditions:
            try:
                result = agent.decide(scenario, condition=cond)
                record = {
                    "scenario_id":         sid,
                    "scenario_type":       stype,
                    "ground_truth":        scenario.get("ground_truth", ""),
                    "is_multi_constraint": scenario.get("is_multi_constraint", False),
                    **result,
                }
            except Exception as exc:
                print(f"  ERROR {sid[:12]} [{cond}]: {exc}")
                record = {
                    "scenario_id":         sid,
                    "scenario_type":       stype,
                    "ground_truth":        scenario.get("ground_truth", ""),
                    "is_multi_constraint": scenario.get("is_multi_constraint", False),
                    "decision":            "error",
                    "classification":      "error",
                    "trace_rate":          0.0,
                    "latency_ms":          0.0,
                    "latency_retrieval_ms":0.0,
                    "latency_inference_ms":0.0,
                    "model_used":          "error",
                    "tamper_alerts":       0,
                    "condition":           cond,
                    "llm_output":          str(exc),
                    "violations":          [],
                }

            _append_result(raw_path, record)
            done += 1
            gt  = record["ground_truth"]
            dec = record.get("strict_decision", record["decision"])
            print(f"  [{done:3d}/{total}] {sid[:16]}  [{stype:8s}] [{cond}]  gt={gt:<30} dec={dec}")

    print(f"\nDone. {done} records written to {raw_path}")


if __name__ == "__main__":
    main()
