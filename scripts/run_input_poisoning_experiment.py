"""
run_input_poisoning_experiment.py

Adversarial Supplier Report Falsification + Local DB Poisoning across all
four conditions (C0/C1/C2/C3).

Threat model
────────────
The attacker compromises (a) the supplier→factory data interface and
(b) the local supply RAG database.  The attacker cannot mint entries on
PureChain (immutable PoA² consensus).

For p% of scenarios the attacker:
  1. Falsifies the current asset state header (soc_pct / temperature_c
     / degradation_pct) toward attacker-favorable values.
  2. Tampers the corresponding asset_records / routing_records used as
     RAG context, with fabricated chain_tx_hash that will fail PureChain
     verification.

Each scenario is then evaluated under all four conditions:

  C0  rule engine on (potentially poisoned) header     → no defense
  C1  LLM on (potentially poisoned) header, empty RAG  → no defense
  C2  LLM + tampered RAG (no hash check)               → silently deceived
  C3  LLM + on-chain hash verification                 → detects tampering;
                                                         falls back to
                                                         PureChain-canonical
                                                         records (recovery)

Hypothesis: as p increases, C0/C1/C2 accuracy collapses; C3 stays flat
because the verification step recovers the genuine record set.

Usage:
    python scripts/run_input_poisoning_experiment.py
    python scripts/run_input_poisoning_experiment.py --rates 0 0.15 0.30 0.50
    python scripts/run_input_poisoning_experiment.py --limit 20    # smoke
    python scripts/run_input_poisoning_experiment.py --resume
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from tqdm import tqdm

from src.logger_setup import setup_logger
from src.layer2.serialiser import RecordSerialiser
from src.layer2.purechain_client import PureChainClient
from src.layer3.llm_agent import LLMAgent, _build_prompt
from src.layer3.physics_checker import PhysicsChecker
from src.evaluation.conditions import _c0_decide

setup_logger("input_poisoning")

RESULTS_DIR  = Path("results")
OUT_FILE     = RESULTS_DIR / "input_poisoning_results.json"
SUMMARY_FILE = RESULTS_DIR / "input_poisoning_summary.json"
FAKE_HASH    = "deadbeef" + "f" * 56

DEFAULT_RATES = [0.0, 0.15, 0.30, 0.50]
SEED = 42

_serialiser = RecordSerialiser()
_checker    = PhysicsChecker()


# ── Ground-truth normaliser ────────────────────────────────────

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
    "hold":     "hold",
    "uncertain":"uncertain",
    "error":    "error",
}

def _norm(label: str) -> str:
    return _GT_MAP.get(str(label).lower().strip(), str(label).lower().strip())


# ── Deterministic poison-set selection ─────────────────────────

def _is_poisoned(scenario_id: str, rate: float, seed: int = SEED) -> bool:
    if rate <= 0:
        return False
    if rate >= 1.0:
        return True
    h = hashlib.sha256(f"{seed}:{scenario_id}".encode()).digest()
    u = int.from_bytes(h[:4], "big") / 2**32
    return u < rate


def _attack_type(scenario_id: str, alpha: float, seed: int = SEED) -> str:
    """
    Attack vector chosen by the adversary for this poisoned scenario.

    Returns
    -------
    'fake_hash'   : cheap attack — fabricated chain_tx_hash that does not
                    exist on PureChain.  C3's verify_tx_hash() returns
                    False, recovery via canonical record set is triggered.
    'forged_hash' : sophisticated attack — keeps the genuine on-chain
                    chain_tx_hash but substitutes tampered payload
                    (e.g. compromised supplier signing key or replay of
                    a stale-but-valid record).  C3's hash-existence
                    check passes — recovery does NOT trigger and the
                    LLM consumes the tampered content.

    alpha = P(fake_hash).  alpha=1 is the original (cheap-only) model;
    alpha=0 is a fully sophisticated adversary; alpha=0.5 is balanced.
    """
    if alpha >= 1.0:
        return "fake_hash"
    if alpha <= 0.0:
        return "forged_hash"
    h = hashlib.sha256(f"{seed}:attack:{scenario_id}".encode()).digest()
    u = int.from_bytes(h[:4], "big") / 2**32
    return "fake_hash" if u < alpha else "forged_hash"


# ── Header / RAG poisoning ─────────────────────────────────────

def _poison_scenario(
    scenario:    Dict[str, Any],
    attack_type: str = "fake_hash",
) -> Dict[str, Any]:
    """
    Return a deep-copied scenario with header AND RAG records tampered.
    The header is poisoned for both attack types; the chain_tx_hash field
    is replaced with FAKE_HASH only under the cheap attack, leaving the
    on-chain hash intact under the sophisticated forged-hash attack.
    """
    s = copy.deepcopy(scenario)
    stype = s.get("scenario_type", "")

    # Header attack: 'rosy' attacker (everything looks fine)
    if stype == "reorder":
        s["soc_pct"]         = 85.0
        s["degradation_pct"] = 100.0
        s["temperature_c"]   = 20.0
    elif stype in ("reroute", "path"):
        s["temperature_c"]   = 20.0
        s["degradation_pct"] = 100.0
    elif stype == "carrier":
        s["degradation_pct"] = 100.0
        s["temperature_c"]   = 20.0

    # RAG payload tamper (applied under both attack types)
    for rec in s.get("asset_records", []):
        if attack_type == "fake_hash":
            rec["chain_tx_hash"] = FAKE_HASH
        # forged_hash: keep the real on-chain hash, only swap content
        if stype == "reorder":
            rec["soc_pct"]         = 85.0
            rec["degradation_pct"] = 100.0
            rec["temperature_c"]   = 20.0
        elif stype in ("reroute", "path"):
            rec["degradation_pct"] = 100.0
            rec["temperature_c"]   = 20.0
        elif stype == "carrier":
            rec["degradation_pct"] = 100.0
            rec["temperature_c"]   = 20.0

    for rec in s.get("routing_records", []):
        if attack_type == "fake_hash":
            rec["chain_tx_hash"] = FAKE_HASH
        rec["amb_temp_max_c"]     = 15.0
        rec["amb_temp_min_c"]     = 5.0
        rec["carrier_variance_c"] = 1.0
        rec["temp_controlled"]    = True

    s["_poisoned"] = True
    s["_attack_type"] = attack_type
    return s


# ── Serialise records (matches vector_index format) ─────────────

def _serialise_asset(raw: Dict) -> Dict:
    s = _serialiser.serialise_asset(raw)
    meta = dict(raw)
    meta["prompt_text"]   = s["prompt_text"]
    meta["chain_tx_hash"] = raw.get("chain_tx_hash", "")
    meta["record_type"]   = "asset_state"
    return meta

def _serialise_routing(raw: Dict) -> Dict:
    s = _serialiser.serialise_routing(raw)
    meta = dict(raw)
    meta["prompt_text"]   = s["prompt_text"]
    meta["chain_tx_hash"] = raw.get("chain_tx_hash", "")
    meta["record_type"]   = "routing_condition"
    return meta


def _build_context(asset_recs: List[Dict], routing_recs: List[Dict]) -> str:
    if not asset_recs and not routing_recs:
        return ""
    lines = ["=== Retrieved Supply Records (PureChain-verified) ==="]
    for rec in asset_recs:
        lines.append(rec.get("prompt_text", ""))
    for rec in routing_recs:
        lines.append(rec.get("prompt_text", ""))
    lines.append("=== End of Supply Records ===")
    return "\n\n".join(lines)


def _verify_and_filter(
    records: List[Dict],
    client:  PureChainClient,
) -> Tuple[List[Dict], int]:
    verified, n_rejected = [], 0
    for rec in records:
        h = rec.get("chain_tx_hash", "")
        if client.verify_tx_hash(h):
            verified.append(rec)
        else:
            n_rejected += 1
    return verified, n_rejected


# ── Per-condition runner ───────────────────────────────────────

def _run_condition(
    agent:             LLMAgent,
    scenario_orig:     Dict[str, Any],   # canonical (untampered) version
    scenario_poisoned: Dict[str, Any],   # as seen by attacker-compromised data layer
    condition:         str,
    client:            PureChainClient,
) -> Dict[str, Any]:
    """
    Execute one (scenario, condition) trial.  C0/C1/C2 see only the
    (possibly poisoned) view; C3 first verifies hashes against PureChain
    and, on tamper detection, falls back to scenario_orig as the
    canonical record set (the conceptual on-chain recovery step).
    """
    sid     = scenario_poisoned["scenario_id"]
    deg_pos = scenario_poisoned.get("degradation_pct")

    # ── C0: rule engine on (poisoned) header ───────────────────
    if condition == "C0":
        result = _c0_decide(scenario_poisoned)
        result["condition"] = "C0"
        return result

    # Default: agent operates on the poisoned scenario
    scenario_for_agent = scenario_poisoned
    tamper_alerts = 0

    if condition == "C1":
        context = ""
    elif condition == "C2":
        a_recs = [_serialise_asset(r)   for r in scenario_poisoned.get("asset_records", [])]
        r_recs = [_serialise_routing(r) for r in scenario_poisoned.get("routing_records", [])]
        context = _build_context(a_recs, r_recs)
    elif condition == "C3":
        # Real PureChain hash verification on the (potentially tampered) records
        a_recs_pois = [_serialise_asset(r)   for r in scenario_poisoned.get("asset_records", [])]
        r_recs_pois = [_serialise_routing(r) for r in scenario_poisoned.get("routing_records", [])]
        v_a, a_n = _verify_and_filter(a_recs_pois, client)
        v_r, r_n = _verify_and_filter(r_recs_pois, client)
        tamper_alerts = a_n + r_n

        if tamper_alerts > 0:
            # Tampering detected → fall back to PureChain-canonical record set
            scenario_for_agent = scenario_orig
            a_recs = [_serialise_asset(r)   for r in scenario_orig.get("asset_records", [])]
            r_recs = [_serialise_routing(r) for r in scenario_orig.get("routing_records", [])]
            context = _build_context(a_recs, r_recs)
        else:
            context = _build_context(v_a, v_r)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    deg_use = scenario_for_agent.get("degradation_pct", deg_pos)
    prompt  = _build_prompt(scenario_for_agent, context, prompt_variant="v1")

    t_inf = time.perf_counter()
    try:
        if agent._is_openai_model():
            llm_output = agent._call_openai(prompt)
        else:
            llm_output = agent._call_ollama(prompt)
    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        llm_output = ""
    latency_inference_ms = (time.perf_counter() - t_inf) * 1000

    rag_mock = {
        "asset_records":   [],
        "routing_records": [],
        "prompt_context":  context,
        "tamper_alerts":   tamper_alerts,
    }
    check = _checker.check(llm_output, rag_mock, degradation_pct=deg_use)

    return {
        "condition":            condition,
        "decision":             check.llm_action or "uncertain",
        "classification":       check.classification,
        "trace_rate":           round(check.trace_rate, 4),
        "latency_ms":           round(latency_inference_ms, 1),
        "latency_retrieval_ms": 0.0,
        "latency_inference_ms": round(latency_inference_ms, 1),
        "model_used":           agent.model,
        "tamper_alerts":        tamper_alerts,
        "llm_output":           llm_output,
        "violations":           check.violations,
    }


# ── Checkpoint helpers ─────────────────────────────────────────

def _load(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)

def _save(path: Path, records: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

def _done_set(records: List[Dict]) -> set:
    return {(r["scenario_id"], r["condition"], r["rate"]) for r in records}


# ── Main runner ────────────────────────────────────────────────

def run_study(
    scenarios:  List[Dict[str, Any]],
    rates:      List[float],
    conditions: List[str],
    agent:      LLMAgent,
    client:     PureChainClient,
    out_path:   Path  = OUT_FILE,
    alpha:      float = 1.0,
    resume:     bool  = True,
) -> List[Dict[str, Any]]:
    existing = _load(out_path) if resume else []
    done     = _done_set(existing)

    total = len(scenarios) * len(conditions) * len(rates)
    bar = tqdm(total=total, desc="poisoning", unit="run", dynamic_ncols=True)

    for rate in rates:
        for scenario in scenarios:
            sid = scenario["scenario_id"]
            poisoned = _is_poisoned(sid, rate)
            atype    = _attack_type(sid, alpha) if poisoned else "none"
            s_pois   = (_poison_scenario(scenario, attack_type=atype)
                        if poisoned else copy.deepcopy(scenario))

            for cond in conditions:
                bar.update(1)
                key = (sid, cond, rate)
                if key in done:
                    bar.set_postfix(skip="OK", rate=rate, cond=cond)
                    continue

                try:
                    r = _run_condition(agent, scenario, s_pois, cond, client)
                except Exception as exc:
                    logger.error(f"{cond}@rate={rate} {sid[:8]}: {exc}")
                    r = {
                        "condition": cond,
                        "decision": "error", "classification": "error",
                        "trace_rate": 0.0, "latency_ms": 0.0,
                        "latency_retrieval_ms": 0.0, "latency_inference_ms": 0.0,
                        "model_used": "error", "tamper_alerts": 0,
                        "llm_output": str(exc), "violations": [],
                    }

                record = {
                    "scenario_id":         sid,
                    "scenario_type":       scenario.get("scenario_type", ""),
                    "ground_truth":        scenario.get("ground_truth", ""),
                    "is_multi_constraint": scenario.get("is_multi_constraint", False),
                    "rate":                rate,
                    "poisoned":            poisoned,
                    "attack_type":         atype,
                    **r,
                }

                existing.append(record)
                done.add(key)
                _save(out_path, existing)

                correct = _norm(record["ground_truth"]) == _norm(record["decision"])
                bar.set_postfix(
                    rate=rate, cond=cond,
                    pois="Y" if poisoned else "N",
                    acc="OK" if correct else "X",
                )

    bar.close()
    return existing


# ── Summary table ──────────────────────────────────────────────

def summarise(
    records:       List[Dict],
    rates:         List[float],
    summary_path:  Path = SUMMARY_FILE,
) -> Dict[str, Any]:
    conditions = ["C0", "C1", "C2", "C3"]
    summary: Dict[str, Any] = {"by_rate": {}, "by_rate_poisoned_only": {}}

    print("\n" + "=" * 76)
    print("  Adversarial Supplier Report Falsification — Accuracy vs Corruption")
    print("=" * 76)
    print(f"  {'rate':<6} " + "  ".join(f"{c:>9}" for c in conditions))
    print("  " + "-" * 60)

    for rate in rates:
        row_acc, row_n = {}, {}
        for cond in conditions:
            recs = [r for r in records
                    if r["rate"] == rate and r["condition"] == cond
                    and r.get("classification") != "error"]
            if not recs:
                row_acc[cond] = None
                row_n[cond]   = 0
                continue
            correct = sum(1 for r in recs
                          if _norm(r["ground_truth"]) == _norm(r["decision"]))
            row_acc[cond] = round(correct / len(recs), 4)
            row_n[cond]   = len(recs)

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
            recs = [r for r in records
                    if r["rate"] == rate and r["condition"] == cond
                    and r.get("poisoned") and r.get("classification") != "error"]
            if not recs:
                row_acc[cond] = None
                row_n[cond]   = 0
                continue
            correct = sum(1 for r in recs
                          if _norm(r["ground_truth"]) == _norm(r["decision"]))
            row_acc[cond] = round(correct / len(recs), 4)
            row_n[cond]   = len(recs)

        print(f"  {rate:<6.2f} " + "  ".join(
            f"{row_acc[c]:>9.3f}" if row_acc[c] is not None else "       NA"
            for c in conditions
        ))
        summary["by_rate_poisoned_only"][f"{rate:.2f}"] = {"accuracy": row_acc, "n": row_n}

    # Per-attack-type breakdown for poisoned scenarios (C3 perspective)
    atypes = sorted({r.get("attack_type", "none") for r in records
                     if r.get("poisoned")})
    if atypes:
        print("\n  C3 accuracy on poisoned scenarios, broken down by attack type:")
        print(f"  {'rate':<6} " + "  ".join(f"{a:>14}" for a in atypes))
        print("  " + "-" * 60)
        atype_summary: Dict[str, Any] = {}
        for rate in rates:
            row, sub = [f"  {rate:<6.2f} "], {}
            for a in atypes:
                rs = [r for r in records
                      if r["rate"] == rate and r["condition"] == "C3"
                      and r.get("poisoned") and r.get("attack_type") == a
                      and r.get("classification") != "error"]
                if not rs:
                    row.append(f"{'NA':>14}")
                    sub[a] = {"accuracy": None, "n": 0}
                    continue
                correct = sum(1 for r in rs
                              if _norm(r["ground_truth"]) == _norm(r["decision"]))
                acc = round(correct / len(rs), 4)
                row.append(f"{acc:>10.3f} (n={len(rs)})")
                sub[a] = {"accuracy": acc, "n": len(rs)}
            print("  ".join(row))
            atype_summary[f"{rate:.2f}"] = sub
        summary["c3_by_rate_attack_type"] = atype_summary

    print("=" * 76)
    _save(summary_path, summary)
    print(f"  Saved -> {summary_path}")
    return summary


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates",      nargs="+", type=float, default=DEFAULT_RATES)
    parser.add_argument("--conditions", nargs="+", default=["C0","C1","C2","C3"])
    parser.add_argument("--limit",      type=int,  default=None)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    with open("data/synthetic/scenarios.json") as f:
        scenarios = json.load(f)
    if args.limit:
        scenarios = scenarios[: args.limit]

    print(f"Input Poisoning Study  —  {len(scenarios)} scenarios × "
          f"{len(args.conditions)} conditions × {len(args.rates)} rates "
          f"= {len(scenarios)*len(args.conditions)*len(args.rates)} runs")

    print("Connecting to PureChain...")
    client = PureChainClient()
    client.connect()

    print("Loading LLM agent...")
    agent = LLMAgent()

    records = run_study(
        scenarios, args.rates, args.conditions, agent, client, resume=args.resume,
    )
    summarise(records, args.rates)


if __name__ == "__main__":
    main()
