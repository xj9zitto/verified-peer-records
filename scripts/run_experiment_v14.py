"""
run_experiment_v14.py

v14 hybrid (predict + decide) experiment runner.  Mirrors run_experiment.py
but dispatches to llm_agent_v14.decide_v14() for each (scenario, condition)
trial, working over data/synthetic/v14_scenarios.json.

Usage:
    python scripts/run_experiment_v14.py
    python scripts/run_experiment_v14.py --limit 5
    python scripts/run_experiment_v14.py --conditions C0 C3
    python scripts/run_experiment_v14.py --resume
    python scripts/run_experiment_v14.py --metrics-only
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from tqdm import tqdm

from src.logger_setup import setup_logger
from src.layer3.llm_agent import LLMAgent
from src.layer3.llm_agent_v14 import decide_v14, V14_ACTION_TOKENS

setup_logger("experiment_v14")

SCEN_FILE = Path("data/synthetic/v14_scenarios.json")
RAW_FILE  = Path("results/v14_raw_results.json")
SUMMARY_FILE = Path("results/v14_summary.json")
CONDITIONS = ["C0", "C1", "C2", "C3"]


def _load(p: Path) -> list:
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _save(p: Path, recs: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(recs, indent=2))


def _done_set(recs: list) -> set:
    return {(r["scenario_id"], r["condition"]) for r in recs}


def run_v14(
    scenarios: list,
    conditions: list,
    agent: LLMAgent,
    output: Path = RAW_FILE,
    resume: bool = True,
    include_outcome: bool = True,
) -> list:
    existing = _load(output) if resume else []
    done     = _done_set(existing)

    total = len(scenarios) * len(conditions)
    bar = tqdm(total=total - len(done), desc="v14", unit="run", dynamic_ncols=True)

    for sc in scenarios:
        sid = sc["scenario_id"]
        # Canonical-peer set used for C3 recovery on tamper detection
        canonical_peers = sc.get("asset_records", [])
        for cond in conditions:
            if (sid, cond) in done:
                continue
            try:
                r = decide_v14(agent, sc, condition=cond,
                                canonical_peers=canonical_peers,
                                include_outcome=include_outcome)
            except Exception as exc:
                logger.error(f"FAILED {sid[:8]} [{cond}]: {exc}")
                r = {
                    "decision": "error", "predicted_stage": None,
                    "predicted_p300": None,
                    "classification": "error", "trace_rate": 0.0,
                    "latency_ms": 0.0, "latency_retrieval_ms": 0.0,
                    "latency_inference_ms": 0.0, "model_used": "error",
                    "tamper_alerts": 0, "condition": cond,
                    "llm_output": str(exc), "violations": [],
                }
            record = {
                "scenario_id":         sid,
                "scenario_type":       sc.get("scenario_type", ""),
                "ground_truth":        sc.get("ground_truth", ""),
                "v14_scenario":        sc.get("metadata", {}).get("v14_scenario", ""),
                "cause_family":        sc.get("metadata", {}).get("cause_family", -1),
                "true_failure_stage":  sc.get("metadata", {}).get("failure_stage"),
                "true_p300":           sc.get("metadata", {}).get("prevention_label_300s"),
                **r,
            }
            existing.append(record)
            done.add((sid, cond))
            _save(output, existing)
            bar.update(1)
            bar.set_postfix(
                cond=cond,
                gt=sc["ground_truth"][:6],
                pred=str(record["decision"])[:6],
                tau=f"{record['trace_rate']:.2f}",
            )

    bar.close()
    return existing


def summarise(records: list) -> dict:
    from collections import Counter, defaultdict
    rates = {}
    print("\n" + "=" * 70)
    print("  v14 hybrid (predict+decide) — summary")
    print("=" * 70)
    print(f"  {'Cond':<5} {'n':>4} {'action_acc':>11} {'stage_acc':>10} {'p300_acc':>9}")
    print("  " + "-" * 56)

    summary = {"by_condition": {}, "by_action_class": {}}
    for cond in ["C0", "C1", "C2", "C3"]:
        recs = [r for r in records
                if r["condition"] == cond and r["classification"] != "error"]
        if not recs:
            continue
        n = len(recs)
        act_correct  = sum(1 for r in recs if r["decision"] == r["ground_truth"])
        stage_recs   = [r for r in recs if r.get("predicted_stage") is not None
                        and r.get("true_failure_stage") is not None]
        stage_correct = sum(
            1 for r in stage_recs
            if int(r["predicted_stage"]) == int(r["true_failure_stage"])
        )
        p300_recs    = [r for r in recs if r.get("predicted_p300") is not None
                        and r.get("true_p300") is not None]
        p300_correct = sum(
            1 for r in p300_recs
            if int(r["predicted_p300"]) == int(r["true_p300"])
        )
        action_acc = act_correct / n
        stage_acc  = stage_correct / len(stage_recs) if stage_recs else None
        p300_acc   = p300_correct / len(p300_recs) if p300_recs else None
        print(f"  {cond:<5} {n:>4} {action_acc:>11.3f} "
              f"{stage_acc if stage_acc is not None else 'NA':>10} "
              f"{p300_acc  if p300_acc  is not None else 'NA':>9}")
        summary["by_condition"][cond] = {
            "n": n,
            "action_acc": round(action_acc, 4),
            "stage_acc":  None if stage_acc is None else round(stage_acc, 4),
            "p300_acc":   None if p300_acc  is None else round(p300_acc, 4),
            "stage_n":    len(stage_recs),
            "p300_n":     len(p300_recs),
        }

    # Per-action-class breakdown for C3
    print("\n  Per-action accuracy (C3):")
    print(f"  {'GT action':<24} {'n':>4} {'acc':>6}")
    by_act = defaultdict(lambda: [0, 0])
    for r in records:
        if r["condition"] != "C3" or r["classification"] == "error":
            continue
        gt = r["ground_truth"]
        by_act[gt][1] += 1
        if r["decision"] == gt:
            by_act[gt][0] += 1
    for gt in sorted(by_act):
        c, t = by_act[gt]
        print(f"  {gt:<24} {t:>4} {c/t:>6.3f}")
        summary["by_action_class"][gt] = {"n": t, "C3_correct": c,
                                          "C3_acc": round(c/t, 4)}

    print("=" * 70)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(f"  Saved -> {SUMMARY_FILE}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",      type=int, default=None)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS,
                    choices=CONDITIONS, metavar="COND")
    ap.add_argument("--scenarios",  type=Path, default=SCEN_FILE)
    ap.add_argument("--output",     type=Path, default=RAW_FILE)
    ap.add_argument("--resume",     action="store_true")
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--no-outcomes", action="store_true",
                    help="State-only ablation: hide peer outcomes from the LLM "
                         "(used to isolate the effect of outcome-bearing peer "
                         "evidence from verification alone).")
    args = ap.parse_args()

    if not args.metrics_only:
        scenarios = json.loads(args.scenarios.read_text())
        if args.limit:
            scenarios = scenarios[: args.limit]
        agent = LLMAgent()
        records = run_v14(scenarios, args.conditions, agent,
                          output=args.output, resume=args.resume,
                          include_outcome=not args.no_outcomes)
    else:
        records = _load(args.output)

    summarise(records)


if __name__ == "__main__":
    main()
