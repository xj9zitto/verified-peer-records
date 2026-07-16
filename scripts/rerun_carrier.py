"""
rerun_carrier.py
Re-run carrier scenarios for C1/C2/C3 after the physics-block recency fix.

The physics_block is now placed AFTER the RAG section so it has recency
bias and overrides any carrier_id values the LLM reads from RAG records.

Usage:
    python scripts/rerun_carrier.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from src.logger_setup import setup_logger
from src.evaluation.conditions import _append_result, RAW_FILE
from src.layer3.llm_agent import LLMAgent

setup_logger("rerun_carrier")

RESULTS_DIR = Path("results")
STYPE       = "carrier"
CONDITIONS  = ["C1", "C2", "C3"]


def main():
    with open("data/synthetic/scenarios.json") as f:
        all_scenarios = json.load(f)

    carrier_scenarios = [s for s in all_scenarios if s["scenario_type"] == STYPE]
    print(f"Found {len(carrier_scenarios)} carrier scenarios to re-run for {CONDITIONS}\n")

    # Remove existing carrier records for C1/C2/C3 from raw_results.json
    raw_path = RAW_FILE
    if raw_path.exists():
        with open(raw_path) as f:
            existing = json.load(f)

        to_remove  = {
            (r["scenario_id"], r["condition"])
            for r in existing
            if r["scenario_type"] == STYPE and r["condition"] in CONDITIONS
        }
        kept = [
            r for r in existing
            if (r["scenario_id"], r["condition"]) not in to_remove
        ]
        print(f"Removing {len(to_remove)} old carrier C1/C2/C3 records from raw_results.json")
        with open(raw_path, "w") as f:
            json.dump(kept, f, indent=2)
    else:
        print("raw_results.json not found — starting fresh")

    agent = LLMAgent()

    total = len(carrier_scenarios) * len(CONDITIONS)
    done  = 0

    for scenario in carrier_scenarios:
        sid = scenario["scenario_id"]
        for cond in CONDITIONS:
            try:
                result = agent.decide(scenario, condition=cond)
                record = {
                    "scenario_id":         sid,
                    "scenario_type":       scenario.get("scenario_type", ""),
                    "ground_truth":        scenario.get("ground_truth", ""),
                    "is_multi_constraint": scenario.get("is_multi_constraint", False),
                    **result,
                }
            except Exception as exc:
                print(f"  ERROR {sid[:12]} [{cond}]: {exc}")
                record = {
                    "scenario_id":         sid,
                    "scenario_type":       scenario.get("scenario_type", ""),
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
            dec = record["decision"]
            ok  = "✓" if gt == dec else "✗"
            print(f"  [{done:3d}/{total}] {sid[:16]}  [{cond}]  gt={gt:<25}  dec={dec:<25} {ok}")

    print(f"\nDone. {done} records written to {raw_path}")


if __name__ == "__main__":
    main()
