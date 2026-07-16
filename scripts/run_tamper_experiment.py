"""
run_tamper_experiment.py
Metric 10 — Tamper Robustness Study.

Demonstrates the core security contribution of PureChain-RAG (C3):
when supply records are adversarially tampered, C2 (SQL RAG) is
silently deceived, while C3 (blockchain-verified RAG) detects and
rejects corrupted records, maintaining decision accuracy.

Attack model
────────────
An adversary modifies the critical supply record in the vector store
(e.g., inflates SOC from 15% to 85% to suppress a reorder decision)
and replaces its chain_tx_hash with a fabricated value.

  C2 (SQL RAG):   no hash check → receives tampered values
                  → LLM is deceived → wrong supply decision
  C3 (PureChain): fabricated hash fails on-chain verification
                  → record excluded from context
                  → LLM falls back to scenario header + real records
                  → correct supply decision

Corruption levels tested
────────────────────────
  0%       — baseline (reuse raw_results.json)
  targeted — 1 critical record tampered per scenario (~12.5% of k=8)
  high     — all records of the critical type tampered (~37.5–62.5%)

Usage:
    python scripts/run_tamper_experiment.py
    python scripts/run_tamper_experiment.py --limit 20   # smoke test
    python scripts/run_tamper_experiment.py --resume
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

import numpy as np

from src.logger_setup import setup_logger
from src.layer2.serialiser import RecordSerialiser
from src.layer2.purechain_client import PureChainClient
from src.layer3.llm_agent import LLMAgent, _build_prompt, _build_query
from src.layer3.physics_checker import PhysicsChecker

setup_logger("tamper_experiment")

RESULTS_DIR  = Path("results")
FAKE_HASH    = "deadbeef" + "f" * 56   # 64-char hex, guaranteed not on-chain

_serialiser  = RecordSerialiser()
_checker     = PhysicsChecker()
_client      = PureChainClient()
_client.connect()                        # required before verify_tx_hash() calls

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


# ── Prompt-context builder (mirrors VectorIndex._build_prompt_context) ─

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


# ── Record serialisation helpers ───────────────────────────────

def _serialise_asset(raw: Dict) -> Dict:
    """Serialise raw asset dict → metadata dict with prompt_text."""
    s = _serialiser.serialise_asset(raw)
    meta = dict(raw)
    meta["prompt_text"]    = s["prompt_text"]
    meta["chain_tx_hash"]  = raw.get("chain_tx_hash", "")
    meta["record_type"]    = "asset_state"
    return meta

def _serialise_routing(raw: Dict) -> Dict:
    """Serialise raw routing dict → metadata dict with prompt_text."""
    s = _serialiser.serialise_routing(raw)
    meta = dict(raw)
    meta["prompt_text"]   = s["prompt_text"]
    meta["chain_tx_hash"] = raw.get("chain_tx_hash", "")
    meta["record_type"]   = "routing_condition"
    return meta


# ── Adversarial tamper generators ──────────────────────────────

def _tamper_asset(rec: Dict, scenario_type: str) -> Dict:
    """
    Return an adversarially modified copy of an asset state record.
    The chain_tx_hash is replaced with a fabricated value so that
    C3's verify_tx_hash() will fail (not on-chain).
    """
    t = copy.deepcopy(rec)
    t["chain_tx_hash"] = FAKE_HASH

    if scenario_type == "reorder":
        # SOC trigger: real value is < 20%.  Inflate to suppress reorder.
        t["soc_pct"] = 85.0

    elif scenario_type in ("reroute", "path"):
        # Degradation trigger: inflate so T_safe rises, masking a thermal risk.
        t["degradation_pct"] = 100.0
        t["temperature_c"]   = 20.0   # make it look cool

    else:
        # Generic: inflate SOC, cool temperature
        t["soc_pct"]       = 85.0
        t["temperature_c"] = 20.0

    # Re-serialise prompt_text to reflect tampered values
    s = _serialiser.serialise_asset(t)
    t["prompt_text"] = s["prompt_text"]
    return t


def _tamper_routing(rec: Dict, scenario_type: str) -> Dict:
    """
    Return an adversarially modified copy of a routing condition record.
    """
    t = copy.deepcopy(rec)
    t["chain_tx_hash"] = FAKE_HASH

    if scenario_type == "carrier":
        # Carrier selection: attacker makes K1/K2 look thermally safe.
        # Reduce variance to 1°C so any carrier appears viable.
        t["carrier_variance_c"] = 1.0
        t["temp_controlled"]    = True   # falsely marks as controlled

    elif scenario_type in ("reroute", "path"):
        # Route safety: attacker makes a dangerous route look safe.
        t["amb_temp_max_c"]     = 15.0   # cool ambient
        t["carrier_variance_c"] = 1.0    # tiny variance
        t["temp_controlled"]    = True

    else:
        t["amb_temp_max_c"]     = 15.0
        t["carrier_variance_c"] = 1.0

    s = _serialiser.serialise_routing(t)
    t["prompt_text"] = s["prompt_text"]
    return t


# ── Hash verification (C3 mode) ────────────────────────────────

def _verify_and_filter(
    records: List[Dict],
    client:  PureChainClient,
) -> Tuple[List[Dict], int]:
    """
    Call verify_tx_hash() on each record's chain_tx_hash.
    Returns (verified_records, n_rejected).

    Records with a hash that does not exist on-chain (e.g. FAKE_HASH)
    are excluded from the returned list.  n_rejected is the count of
    records that failed verification — this becomes tamper_alerts for C3.
    """
    verified  = []
    n_rejected = 0
    for rec in records:
        h = rec.get("chain_tx_hash", "")
        if client.verify_tx_hash(h):
            verified.append(rec)
        else:
            n_rejected += 1
            logger.debug(f"C3 rejected record with hash {h[:16]}… (not on-chain)")
    return verified, n_rejected


# ── Build C2/C3 record lists for a given corruption scenario ───

def _build_tamper_contexts(
    scenario: Dict,
    level:    str,   # "baseline" | "targeted" | "high"
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], int]:
    """
    Build record lists for C2 and C3 under the given tamper level.

    Returns
    -------
    (c2_asset, c2_routing, c3_candidate_asset, c3_candidate_routing, n_tampered)

    C2 receives the records directly (including tampered ones) — no verification.
    C3 receives *candidate* records; _run_one calls _verify_and_filter() on them
    via live PureChain verify_tx_hash() calls.  Tampered records carry FAKE_HASH
    and are rejected; real records carry valid hashes and pass through.
    n_tampered is the number of records where FAKE_HASH was injected.
    """
    stype = scenario.get("scenario_type", "")

    real_asset   = [_serialise_asset(r)   for r in scenario.get("asset_records",   [])]
    real_routing = [_serialise_routing(r) for r in scenario.get("routing_records", [])]

    if level == "baseline":
        return real_asset, real_routing, real_asset, real_routing, 0

    tampered_asset   = [_tamper_asset(r,   stype) for r in scenario.get("asset_records",   [])]
    tampered_routing = [_tamper_routing(r, stype) for r in scenario.get("routing_records", [])]

    if level == "targeted":
        # Tamper exactly ONE critical record.
        if stype == "reorder" and tampered_asset:
            c2_asset          = [tampered_asset[0]] + real_asset[1:]
            c2_routing        = real_routing
            c3_cand_asset     = [tampered_asset[0]] + real_asset[1:]  # C3 verifies each
            c3_cand_routing   = real_routing
            n = 1
        elif real_routing and tampered_routing:
            c2_asset          = real_asset
            c2_routing        = [tampered_routing[0]] + real_routing[1:]
            c3_cand_asset     = real_asset
            c3_cand_routing   = [tampered_routing[0]] + real_routing[1:]
            n = 1
        else:
            return real_asset, real_routing, real_asset, real_routing, 0

    else:  # "high" — tamper ALL records of the critical type
        if stype == "reorder":
            c2_asset          = tampered_asset
            c2_routing        = real_routing
            c3_cand_asset     = tampered_asset   # C3 will reject all (FAKE_HASH)
            c3_cand_routing   = real_routing
            n = len(tampered_asset)
        else:
            c2_asset          = real_asset
            c2_routing        = tampered_routing
            c3_cand_asset     = real_asset
            c3_cand_routing   = tampered_routing  # C3 will reject all (FAKE_HASH)
            n = len(tampered_routing)

    return c2_asset, c2_routing, c3_cand_asset, c3_cand_routing, n


# ── Run a single LLM call ──────────────────────────────────────

def _run_one(
    agent:           LLMAgent,
    scenario:        Dict,
    asset_records:   List[Dict],    # pre-built serialised records for this condition
    routing_records: List[Dict],
    condition:       str,           # "C2" or "C3"
    n_tampered:      int,
    level:           str,
    client:          PureChainClient,
) -> Dict:
    """
    For C2: build context from asset_records + routing_records as-is.
            No hash verification — tampered records pass through silently.
    For C3: call verify_tx_hash() on each record's chain_tx_hash via live
            PureChain.  Records with FAKE_HASH (not on-chain) are excluded.
            tamper_alerts = count of rejected records.
    """
    deg = scenario.get("degradation_pct")

    if condition == "C3":
        # Live on-chain verification — the security contribution of PureChain
        verified_asset,   alerts_a = _verify_and_filter(asset_records,   client)
        verified_routing, alerts_r = _verify_and_filter(routing_records, client)
        tamper_alerts = alerts_a + alerts_r
        context = _build_context(verified_asset, verified_routing)
        logger.info(
            f"C3 verify: {tamper_alerts} record(s) rejected "
            f"(asset={alerts_a}, routing={alerts_r})"
        )
    else:
        # C2: accept all records without verification
        tamper_alerts = 0
        context = _build_context(asset_records, routing_records)

    prompt = _build_prompt(scenario, context, prompt_variant="v1")

    t0 = time.perf_counter()
    try:
        if agent._is_openai_model():
            llm_output = agent._call_openai(prompt)
        else:
            llm_output = agent._call_ollama(prompt)
    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        llm_output = ""
    latency_ms = (time.perf_counter() - t0) * 1000

    rag_mock = {
        "asset_records":   [],
        "routing_records": [],
        "prompt_context":  context,
        "tamper_alerts":   tamper_alerts,
    }
    check = _checker.check(llm_output, rag_mock, degradation_pct=deg)

    return {
        "scenario_id":         scenario["scenario_id"],
        "scenario_type":       scenario.get("scenario_type", ""),
        "ground_truth":        scenario.get("ground_truth", ""),
        "is_multi_constraint": scenario.get("is_multi_constraint", False),
        "condition":           condition,
        "tamper_level":        level,
        "n_tampered":          n_tampered,
        "tamper_alerts":       tamper_alerts,
        "decision":            check.llm_action or "uncertain",
        "classification":      check.classification,
        "trace_rate":          round(check.trace_rate, 4),
        "latency_ms":          round(latency_ms, 1),
        "llm_output":          llm_output,
        "violations":          check.violations,
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
    return {(r["scenario_id"], r["condition"], r["tamper_level"]) for r in records}


# ── Main experiment runner ─────────────────────────────────────

def run_tamper_study(
    scenarios: List[Dict],
    agent:     LLMAgent,
    resume:    bool = True,
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Run C2 and C3 under targeted and high-corruption tamper attacks.

    Returns {level: {condition: [records]}}.
    """
    out_path = RESULTS_DIR / "tamper_results.json"
    existing = _load(out_path) if resume else []
    done     = _done_set(existing)

    levels     = ["targeted", "high"]
    conditions = ["C2", "C3"]

    total = len(scenarios) * len(levels) * len(conditions)
    bar   = tqdm(total=total, desc="tamper", unit="run", dynamic_ncols=True)

    for level in levels:
        for scenario in scenarios:
            (c2_asset, c2_routing,
             c3_cand_asset, c3_cand_routing,
             n_tampered) = _build_tamper_contexts(scenario, level)

            records_map = {
                "C2": (c2_asset,      c2_routing),
                "C3": (c3_cand_asset, c3_cand_routing),
            }

            for cond in conditions:
                key = (scenario["scenario_id"], cond, level)
                bar.update(1)

                if key in done:
                    bar.set_postfix(skip="✓", level=level, cond=cond)
                    continue

                asset_recs, routing_recs = records_map[cond]
                try:
                    rec = _run_one(
                        agent, scenario,
                        asset_recs, routing_recs,
                        cond, n_tampered, level, _client,
                    )
                except Exception as exc:
                    logger.error(f"{cond}/{level} {scenario['scenario_id'][:8]}: {exc}")
                    rec = {
                        "scenario_id":         scenario["scenario_id"],
                        "scenario_type":       scenario.get("scenario_type", ""),
                        "ground_truth":        scenario.get("ground_truth", ""),
                        "is_multi_constraint": scenario.get("is_multi_constraint", False),
                        "condition":     cond,
                        "tamper_level":  level,
                        "n_tampered":    n_tampered,
                        "tamper_alerts": 0,
                        "decision": "error", "classification": "error",
                        "trace_rate": 0.0, "latency_ms": 0.0,
                        "llm_output": str(exc), "violations": [],
                    }

                existing.append(rec)
                done.add(key)
                _save(out_path, existing)

                correct = _norm(rec["ground_truth"]) == _norm(rec["decision"])
                bar.set_postfix(
                    level=level, cond=cond,
                    acc="✓" if correct else "✗",
                    n=n_tampered,
                )

    bar.close()

    # Organise results
    results = {}
    for level in levels:
        results[level] = {}
        for cond in conditions:
            results[level][cond] = [
                r for r in existing
                if r["tamper_level"] == level
                and r["condition"]   == cond
                and r["classification"] != "error"
            ]

    print(f"\n  Saved → {out_path}")
    return results


# ── Load baseline from raw_results.json ───────────────────────

def load_baseline() -> Dict[str, List[Dict]]:
    path = RESULTS_DIR / "raw_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        all_r = json.load(f)
    return {
        "C2": [r for r in all_r if r["condition"] == "C2" and r["classification"] != "error"],
        "C3": [r for r in all_r if r["condition"] == "C3" and r["classification"] != "error"],
    }


# ── Summary table & plot data ──────────────────────────────────

def summarise(
    baseline:      Dict[str, List[Dict]],
    tamper_results: Dict[str, Dict[str, List[Dict]]],
) -> None:
    conditions = ["C2", "C3"]
    levels_all = ["baseline", "targeted", "high"]

    # Gather all records into one lookup
    all_recs: Dict[str, Dict[str, List[Dict]]] = {
        "baseline": baseline,
        **tamper_results,
    }

    print("\n" + "=" * 70)
    print("  METRIC 10 — Tamper Robustness (Adversarial Supply Record Injection)")
    print("=" * 70)
    print(f"  {'Level':<12}  {'Condition':<6}  {'n':>5}  "
          f"{'Accuracy':>9}  {'Δ vs C3 base':>13}  {'Detect Rate':>11}")
    print("  " + "-" * 64)

    c3_base_acc = None
    summary_out: Dict[str, Any] = {}

    for level in levels_all:
        for cond in conditions:
            recs = all_recs.get(level, {}).get(cond, [])
            if not recs:
                continue
            correct = sum(1 for r in recs
                          if _norm(r["ground_truth"]) == _norm(r["decision"]))
            acc = correct / len(recs)

            if level == "baseline" and cond == "C3":
                c3_base_acc = acc

            delta = f"{acc - c3_base_acc:+.3f}" if c3_base_acc is not None else "  —  "

            # Detection rate: fraction of scenarios where C3 correctly flagged tampers
            if cond == "C3" and level != "baseline":
                detect_rate = np.mean([
                    1 if r["tamper_alerts"] > 0 else 0
                    for r in recs
                ])
                detect_str = f"{detect_rate:.3f}"
            else:
                detect_str = "   —  "

            print(f"  {level:<12}  {cond:<6}  {len(recs):>5}  "
                  f"{acc:>9.3f}  {delta:>13}  {detect_str:>11}")

            key = f"{level}_{cond}"
            summary_out[key] = {
                "n":           len(recs),
                "accuracy":    round(acc, 4),
                "delta_vs_c3_base": round(acc - (c3_base_acc or 0), 4),
            }

    print("=" * 70)

    # Per-type breakdown
    print("\n  Per-type accuracy under targeted tamper:")
    print(f"  {'Type':<10}  {'C2_targeted':>12}  {'C3_targeted':>12}  {'Gap':>8}")
    print("  " + "-" * 46)
    stypes = ["reorder", "reroute", "path", "carrier"]
    for st in stypes:
        c2r = [r for r in all_recs.get("targeted", {}).get("C2", [])
               if r["scenario_type"] == st]
        c3r = [r for r in all_recs.get("targeted", {}).get("C3", [])
               if r["scenario_type"] == st]
        if not c2r or not c3r:
            continue
        a2 = sum(1 for r in c2r if _norm(r["ground_truth"]) == _norm(r["decision"])) / len(c2r)
        a3 = sum(1 for r in c3r if _norm(r["ground_truth"]) == _norm(r["decision"])) / len(c3r)
        print(f"  {st:<10}  {a2:>12.3f}  {a3:>12.3f}  {a3-a2:>+8.3f}")

    # Save JSON summary
    out = RESULTS_DIR / "tamper_summary.json"
    with open(out, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\n  Saved → {out}")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open("data/synthetic/scenarios.json") as f:
        scenarios = json.load(f)
    if args.limit:
        scenarios = scenarios[:args.limit]

    print(f"\nTamper Robustness Study — {len(scenarios)} scenarios")
    print("Loading LLM agent...")
    agent = LLMAgent()

    print("\n=== Running C2/C3 under adversarial tamper ===")
    tamper_results = run_tamper_study(scenarios, agent, resume=args.resume)

    baseline = load_baseline()
    summarise(baseline, tamper_results)


if __name__ == "__main__":
    main()
