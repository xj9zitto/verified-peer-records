"""
run_v14_poisoning_experiment.py

Adversarial observable-channel poisoning of v14 hybrid (predict + decide)
scenarios, across all four conditions (C0/C1/C2/C3).

Threat model
────────────
Attacker compromises the BMS-to-factory data interface AND the local
peer-summary database.  PureChain is immutable.

For p% of scenarios, attacker:
  1. Falsifies the 60s observable window summary (voltage/current/
     temperature/dT/dt/SOC) toward attacker-favorable 'rosy' values
     that suppress urgent / replace actions.
  2. Tampers the verified-peer-trajectory records, with either:
       'fake_hash'  : fabricated chain_tx_hash → C3 detects, recovers
       'forged_hash': real on-chain hash, content tampered → C3
                      hash-existence check passes; LLM consumes
                      tampered peer outcomes

Conditions:
  C0  rule engine on observable summary               → no defense
  C1  LLM on observable summary, no peers             → no defense
  C2  LLM + tampered peers (no hash check)            → silently deceived
  C3  LLM + on-chain-verified peers; on tamper detect →
      fall back to canonical (untampered) scenario

Hypothesis: C0/C1/C2 collapse linearly with p; C3 stays flat (against
fake_hash) and shows a shallow slope (against forged_hash).  Headline
figure: accuracy vs p, with α=0.5 mixed attack.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from tqdm import tqdm

from src.logger_setup import setup_logger
from src.layer3.llm_agent import LLMAgent
from src.layer3.llm_agent_v14 import (
    decide_v14, V14_ACTION_TOKENS, _verify_peer_records,
    build_prompt_v14,
)
from src.layer1.v14_rule_engine import c0_decide_v14

setup_logger("v14_poisoning")

RESULTS_DIR  = Path("results")
SCEN_FILE    = Path("data/synthetic/v14_scenarios.json")
OUT_FILE     = RESULTS_DIR / "v14_poisoning_results.json"
SUMMARY_FILE = RESULTS_DIR / "v14_poisoning_summary.json"
FAKE_HASH    = "deadbeef" + "f" * 56

DEFAULT_RATES = [0.0, 0.15, 0.30, 0.50]
SEED = 42


# ── Deterministic poison-set + attack-type selection ──────────

def _is_poisoned(sid: str, rate: float, seed: int = SEED) -> bool:
    if rate <= 0:
        return False
    if rate >= 1.0:
        return True
    h = hashlib.sha256(f"{seed}:{sid}".encode()).digest()
    u = int.from_bytes(h[:4], "big") / 2**32
    return u < rate


def _attack_type(sid: str, alpha: float, seed: int = SEED) -> str:
    if alpha >= 1.0:
        return "fake_hash"
    if alpha <= 0.0:
        return "forged_hash"
    h = hashlib.sha256(f"{seed}:attack:{sid}".encode()).digest()
    u = int.from_bytes(h[:4], "big") / 2**32
    return "fake_hash" if u < alpha else "forged_hash"


# ── Observable-window poisoning ───────────────────────────────

def _poison_obs_summary(summary: Dict[str, Any]) -> None:
    """In-place rosy poisoning of the 60s window summary."""
    rosy_targets = {
        "voltage_V_measured":     {"max": 3.70, "last": 3.65, "mean": 3.68,
                                   "first": 3.65, "min": 3.65, "trend": "flat"},
        "current_A_measured":     {"max": 0.50, "last": 0.45, "mean": 0.48,
                                   "first": 0.45, "min": 0.40, "trend": "flat"},
        "temperature_C_measured": {"max": 25.0, "last": 24.5, "mean": 24.8,
                                   "first": 24.5, "min": 24.0, "trend": "flat"},
        "SOC_with_drift":         {"max": 0.95, "last": 0.95, "mean": 0.95,
                                   "first": 0.95, "min": 0.94, "trend": "flat"},
        "dT_dt_meas_C_per_s":     {"max": 0.005, "last": 0.001, "mean": 0.002,
                                   "first": 0.001, "min": -0.003, "trend": "flat"},
        "dV_dt_meas_V_per_s":     {"max": 0.001, "last": 0.0, "mean": 0.0001,
                                   "first": 0.0, "min": -0.001, "trend": "flat"},
    }
    for ch, vals in rosy_targets.items():
        if ch in summary:
            summary[ch].update(vals)


def _poison_peer_outcomes(peer: Dict[str, Any]) -> None:
    """Rosy outcome tampering — claim peers stayed safe even if they didn't."""
    out = peer.get("outcome", {})
    out["failure_stage_max"] = 0
    out["did_escalate"]      = False
    out["reached_failure"]   = False


def _poison_scenario(
    scenario:    Dict[str, Any],
    attack_type: str,
) -> Dict[str, Any]:
    s = copy.deepcopy(scenario)

    # Header poisoning: rosy obs_window summary + snapshot
    if "obs_window" in s and "summary" in s["obs_window"]:
        _poison_obs_summary(s["obs_window"]["summary"])
    if "obs_window" in s and "snapshot" in s["obs_window"]:
        s["obs_window"]["snapshot"] = {
            "voltage_V_measured":     3.65,
            "current_A_measured":     0.45,
            "temperature_C_measured": 24.5,
            "SOC_with_drift":         0.95,
            "dT_dt_meas_C_per_s":     0.001,
            "dV_dt_meas_V_per_s":     0.0,
        }

    # Peer-RAG poisoning: tamper outcomes; hash policy depends on attack type
    for peer in s.get("asset_records", []):
        if attack_type == "fake_hash":
            peer["chain_tx_hash"] = FAKE_HASH
        # else forged_hash: keep the real chain_tx_hash, only tamper content
        _poison_peer_outcomes(peer)

    s["_poisoned"]    = True
    s["_attack_type"] = attack_type
    return s


# ── Per-condition runner ──────────────────────────────────────

def _run_condition(
    agent,
    scenario_orig:     Dict[str, Any],
    scenario_poisoned: Dict[str, Any],
    condition:         str,
    include_outcome:   bool = True,
) -> Dict[str, Any]:
    if condition == "C0":
        # Rule engine reads scenario_poisoned (it has no defense)
        result = c0_decide_v14(scenario_poisoned)
        return result

    # C1/C2/C3 — go through decide_v14, which handles C3 hash-verify
    # + canonical fallback against scenario_orig
    canonical_peers = scenario_orig.get("asset_records", [])
    result = decide_v14(
        agent, scenario_poisoned, condition=condition,
        canonical_peers=canonical_peers,
        include_outcome=include_outcome,
    )
    # For C3 with fake_hash, canonical-recovery fallback only restores peers,
    # not the obs_window header.  We additionally restore the header here when
    # tampering is detected, completing the recovery story.
    if condition == "C3" and result.get("tamper_alerts", 0) > 0:
        try:
            result2 = decide_v14(agent, scenario_orig, condition="C3",
                                 canonical_peers=canonical_peers,
                                 include_outcome=include_outcome)
            result2["tamper_alerts"] = result["tamper_alerts"]
            result2["recovered"]     = True
            return result2
        except Exception as exc:
            logger.error(f"C3 recovery re-run failed: {exc}")
    return result


# ── Checkpoint helpers ────────────────────────────────────────

def _load(p: Path) -> list:
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _save(p: Path, recs: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(recs, indent=2))


def _done_set(recs: list) -> set:
    return {(r["scenario_id"], r["condition"], r["rate"]) for r in recs}


# ── GT normaliser ─────────────────────────────────────────────

def _correct(rec: Dict[str, Any]) -> bool:
    return rec.get("decision") == rec.get("ground_truth")


# ── Main study ────────────────────────────────────────────────

def run_study(
    scenarios:  List[Dict[str, Any]],
    rates:      List[float],
    conditions: List[str],
    agent:      LLMAgent,
    alpha:      float = 0.5,
    out_path:   Path  = OUT_FILE,
    resume:     bool  = False,
    include_outcome: bool = True,
) -> List[Dict[str, Any]]:
    existing = _load(out_path) if resume else []
    done     = _done_set(existing)

    total = len(scenarios) * len(conditions) * len(rates)
    bar = tqdm(total=total, desc="v14_poison", unit="run", dynamic_ncols=True)

    for rate in rates:
        for sc in scenarios:
            sid      = sc["scenario_id"]
            poisoned = _is_poisoned(sid, rate)
            atype    = _attack_type(sid, alpha) if poisoned else "none"
            s_pois   = (_poison_scenario(sc, atype) if poisoned
                        else copy.deepcopy(sc))

            for cond in conditions:
                bar.update(1)
                key = (sid, cond, rate)
                if key in done:
                    bar.set_postfix(skip="OK", rate=rate, cond=cond)
                    continue

                try:
                    r = _run_condition(agent, sc, s_pois, cond,
                                       include_outcome=include_outcome)
                except Exception as exc:
                    logger.error(f"{cond}@{rate} {sid[:8]}: {exc}")
                    r = {
                        "decision": "error", "predicted_stage": None,
                        "predicted_p300": None, "classification": "error",
                        "trace_rate": 0.0, "latency_ms": 0.0,
                        "latency_retrieval_ms": 0.0, "latency_inference_ms": 0.0,
                        "model_used": "error", "tamper_alerts": 0,
                        "condition": cond, "llm_output": str(exc),
                        "violations": [],
                    }

                record = {
                    "scenario_id":         sid,
                    "scenario_type":       sc.get("scenario_type", ""),
                    "ground_truth":        sc.get("ground_truth", ""),
                    "v14_scenario":        sc.get("metadata", {}).get("v14_scenario", ""),
                    "cause_family":        sc.get("metadata", {}).get("cause_family", -1),
                    "true_failure_stage":  sc.get("metadata", {}).get("failure_stage"),
                    "true_p300":           sc.get("metadata", {}).get("prevention_label_300s"),
                    "rate":                rate,
                    "poisoned":            poisoned,
                    "attack_type":         atype,
                    **r,
                }
                existing.append(record)
                done.add(key)
                _save(out_path, existing)

                ok = "OK" if _correct(record) else "X"
                bar.set_postfix(rate=rate, cond=cond,
                                pois="Y" if poisoned else "N", acc=ok)

    bar.close()
    return existing


def summarise(records: list, rates: list,
              summary_path: Path = SUMMARY_FILE) -> dict:
    conditions = sorted({r["condition"] for r in records})
    summary = {"by_rate": {}, "by_rate_poisoned_only": {},
               "by_attack_type": {}}

    print("\n" + "=" * 76)
    print("  v14 input-poisoning — accuracy vs corruption rate (alpha=0.5)")
    print("=" * 76)
    print(f"  {'rate':<6} " + "  ".join(f"{c:>9}" for c in conditions))
    print("  " + "-" * 60)

    for rate in rates:
        row_acc, row_n = {}, {}
        for cond in conditions:
            rs = [r for r in records
                  if r["rate"] == rate and r["condition"] == cond
                  and r.get("classification") != "error"]
            n = len(rs)
            c = sum(1 for r in rs if _correct(r))
            row_acc[cond] = round(c/n, 4) if n else None
            row_n[cond]   = n
        print(f"  {rate:<6.2f} " + "  ".join(
            f"{row_acc[c]:>9.3f}" if row_acc[c] is not None else "       NA"
            for c in conditions
        ))
        summary["by_rate"][f"{rate:.2f}"] = {"accuracy": row_acc, "n": row_n}

    print("\n  Poisoned scenarios only (attack effectiveness):")
    print(f"  {'rate':<6} " + "  ".join(f"{c:>9}" for c in conditions))
    print("  " + "-" * 60)
    for rate in rates:
        row_acc, row_n = {}, {}
        for cond in conditions:
            rs = [r for r in records
                  if r["rate"] == rate and r["condition"] == cond
                  and r.get("poisoned")
                  and r.get("classification") != "error"]
            n = len(rs)
            c = sum(1 for r in rs if _correct(r))
            row_acc[cond] = round(c/n, 4) if n else None
            row_n[cond]   = n
        print(f"  {rate:<6.2f} " + "  ".join(
            f"{row_acc[c]:>9.3f}" if row_acc[c] is not None else "       NA"
            for c in conditions
        ))
        summary["by_rate_poisoned_only"][f"{rate:.2f}"] = {
            "accuracy": row_acc, "n": row_n}

    # Verified conditions broken down by attack type
    for vcond in [c for c in conditions if c in ("C3", "C4")]:
        print(f"\n  {vcond} by attack type (poisoned scenarios only):")
        print(f"  {'rate':<6}   {'fake_hash':>14}  {'forged_hash':>14}")
        print("  " + "-" * 50)
        summary["by_attack_type"][vcond] = {}
        for rate in rates:
            if rate <= 0:
                continue
            row, sub = [f"  {rate:<6.2f} "], {}
            for atype in ("fake_hash", "forged_hash"):
                rs = [r for r in records
                      if r["rate"] == rate and r["condition"] == vcond
                      and r.get("attack_type") == atype
                      and r.get("classification") != "error"]
                n = len(rs)
                c = sum(1 for r in rs if _correct(r))
                acc = round(c/n, 4) if n else None
                row.append(f"{acc:>10.3f} (n={n})" if acc is not None else f"{'NA':>14}")
                sub[atype] = {"accuracy": acc, "n": n}
            print("  ".join(row))
            summary["by_attack_type"][vcond][f"{rate:.2f}"] = sub
    print("=" * 76)

    _save(summary_path, summary)
    print(f"  Saved -> {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates",      nargs="+", type=float, default=DEFAULT_RATES)
    ap.add_argument("--conditions", nargs="+", default=["C0", "C1", "C2", "C3"])
    ap.add_argument("--alpha",      type=float, default=0.5)
    ap.add_argument("--limit",      type=int,   default=None)
    ap.add_argument("--out",        type=Path,  default=OUT_FILE)
    ap.add_argument("--summary-out", type=Path, default=SUMMARY_FILE,
                    help="Path to write the per-rate summary JSON.")
    ap.add_argument("--scenarios",  type=Path,  default=SCEN_FILE)
    ap.add_argument("--resume",     action="store_true")
    ap.add_argument("--no-outcomes", action="store_true",
                    help="State-only ablation: hide peer outcomes from the LLM "
                         "(used to isolate the effect of outcome-bearing peer "
                         "evidence from verification alone).")
    ap.add_argument("--shuffle-outcomes", action="store_true",
                    help="Shuffled-outcome control: permute peer outcome "
                         "fields across the whole corpus (seeded), breaking "
                         "the state-outcome association while preserving the "
                         "outcome distribution and prompt structure.")
    args = ap.parse_args()

    scenarios = json.loads(args.scenarios.read_text())
    if args.limit:
        scenarios = scenarios[: args.limit]

    if args.shuffle_outcomes:
        import random as _random
        rng = _random.Random(SEED)
        all_out = [p["outcome"] for sc in scenarios
                   for p in sc.get("asset_records", [])]
        shuffled = all_out[:]
        rng.shuffle(shuffled)
        i = 0
        for sc in scenarios:
            for p in sc.get("asset_records", []):
                p["outcome"] = shuffled[i]
                i += 1
        print(f"  SHUFFLED-OUTCOME CONTROL: {i} peer outcomes permuted "
              f"across the corpus (seed={SEED})")

    print(f"v14 Input Poisoning  —  {len(scenarios)} scenarios x "
          f"{len(args.conditions)} conditions x {len(args.rates)} rates  "
          f"= {len(scenarios)*len(args.conditions)*len(args.rates)} runs")
    print(f"  alpha = {args.alpha}  output -> {args.out}")
    if args.no_outcomes:
        print("  STATE-ONLY ABLATION: peer outcomes hidden from LLM")

    agent = LLMAgent()
    records = run_study(scenarios, args.rates, args.conditions, agent,
                        alpha=args.alpha, out_path=args.out, resume=args.resume,
                        include_outcome=not args.no_outcomes)
    summarise(records, args.rates, summary_path=args.summary_out)


if __name__ == "__main__":
    main()
