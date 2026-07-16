"""
run_experiment.py
Top-level experiment runner — 120 scenarios × 4 conditions.

Usage:
    python scripts/run_experiment.py               # full run
    python scripts/run_experiment.py --resume      # skip already-done rows
    python scripts/run_experiment.py --conditions C0 C3  # subset
    python scripts/run_experiment.py --limit 5     # first N scenarios (smoke)
    python scripts/run_experiment.py --metrics-only # recompute metrics only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from src.logger_setup import setup_logger
from src.evaluation.conditions import run, RAW_FILE, CONDITIONS
from src.evaluation.metrics import compute

setup_logger("experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Factory LLM Experiment")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip (scenario, condition) pairs already in output")
    parser.add_argument("--conditions",   nargs="+", default=CONDITIONS,
                        choices=CONDITIONS, metavar="COND",
                        help="Conditions to run (default: C0 C1 C2 C3)")
    parser.add_argument("--types",        nargs="+", default=None,
                        choices=["reorder", "reroute", "path", "carrier"],
                        metavar="TYPE",
                        help="Scenario types to run (default: all)")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Run only the first N scenarios (for smoke testing)")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Skip inference, recompute metrics from existing raw_results.json")
    parser.add_argument("--output",       type=Path, default=RAW_FILE,
                        help=f"Path to output JSON (default: {RAW_FILE})")
    args = parser.parse_args()

    with open("data/synthetic/scenarios.json") as f:
        scenarios = json.load(f)

    if args.types:
        scenarios = [s for s in scenarios if s["scenario_type"] in args.types]

    if args.limit:
        scenarios = scenarios[: args.limit]

    if not args.metrics_only:
        from src.layer3.llm_agent import LLMAgent
        agent = LLMAgent()

        results = run(
            scenarios=scenarios,
            conditions=args.conditions,
            agent=agent,
            output=args.output,
            resume=args.resume,
        )
    else:
        if not args.output.exists():
            print(f"No results file found at {args.output}. Run without --metrics-only first.")
            sys.exit(1)
        with open(args.output) as f:
            results = json.load(f)

    compute(results, save=True, verbose=True)


if __name__ == "__main__":
    main()
