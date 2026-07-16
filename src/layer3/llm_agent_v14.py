"""
llm_agent_v14.py
Layer 3 — v14 hybrid (predict + decide) prompt builder, output parser,
and decision pipeline.

For scenario_type == "v14_safety", the LLM is asked to:
  1. Predict failure_stage and prevention_300s from observable channels
  2. Choose ONE supply action from a 5-item menu

The structured prompt presents a 60s observable window summary and a set
of verified peer-trajectory records (with outcomes), all anchored to
PureChain via chain_tx_hash.  C2 receives unverified records (potentially
tampered); C3 receives only on-chain-verified records, with canonical
recovery on tamper detection.
"""

from __future__ import annotations
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.layer2.v14_content import peer_content_digest, peer_key

from src.layer1.v14_rule_engine import (
    gt_action_from, c0_decide_v14,
)


# ── Action menu ───────────────────────────────────────────────

V14_ACTIONS = [
    ("safe",     "continue or monitor; no Stage>=2 risk in near term"),
    ("replace",  "schedule replacement battery; warning state with Stage>=2 imminent"),
    ("urgent",   "take asset offline NOW; Stage>=2 already in progress"),
]

V14_ACTION_TOKENS = {a for a, _ in V14_ACTIONS}


# ── Output parsing ────────────────────────────────────────────

_ACTION_LINE_RE = re.compile(
    r"^\s*(?:final\s+)?action\s*[:=\-]\s*([a-z_]+)\b",
    re.IGNORECASE | re.MULTILINE,
)

_STAGE_LINE_RE = re.compile(
    r"(?:predicted\s+)?(?:failure\s*[_ ]?)?stage\s*[:=\-]\s*([0-3])\b",
    re.IGNORECASE,
)

_PREVENTION_LINE_RE = re.compile(
    r"(?:stage\s*-?\s*2\s*in\s*5\s*min|prevention(?:_300s)?)\s*[:=\-]\s*"
    r"(yes|y|no|n|true|false|1|0)\b",
    re.IGNORECASE,
)


def parse_v14_output(text: str) -> Dict[str, Any]:
    """
    Extract (predicted_stage, predicted_p300, action) from an LLM response.
    Falls back to keyword search if structured fields aren't found.
    Returns a dict with keys: action, predicted_stage, predicted_p300.
    """
    out = {"action": None, "predicted_stage": None, "predicted_p300": None}

    # Action — structured field first, then keyword fallback
    m = _ACTION_LINE_RE.search(text)
    if m:
        cand = m.group(1).strip().lower()
        if cand in V14_ACTION_TOKENS:
            out["action"] = cand

    if out["action"] is None:
        lower = text.lower()
        # Order matters: prefer the most-specific action token if multiple appear
        for tok in ["urgent", "replace", "safe"]:
            if tok in lower:
                out["action"] = tok
                break

    # Stage
    m = _STAGE_LINE_RE.search(text)
    if m:
        try:
            out["predicted_stage"] = int(m.group(1))
        except ValueError:
            pass

    # Prevention
    m = _PREVENTION_LINE_RE.search(text)
    if m:
        v = m.group(1).strip().lower()
        out["predicted_p300"] = 1 if v in ("yes", "y", "true", "1") else 0

    return out


# ── Context builders ──────────────────────────────────────────

def _format_obs_window(obs_window: Dict[str, Any]) -> str:
    """Render the observable summary block."""
    summary = obs_window.get("summary", {})
    if not summary:
        return "(no observable window available)"

    lines = [
        f"Window: t={obs_window.get('t_start_s', 0):.0f}s "
        f"to t={obs_window.get('t_end_s', 0):.0f}s "
        f"({obs_window.get('n_timesteps', 0)} timesteps)"
    ]
    pretty_names = {
        "voltage_V_measured":     "voltage [V]",
        "current_A_measured":     "current [A]",
        "temperature_C_measured": "temperature [°C]",
        "SOC_with_drift":         "SOC",
        "dT_dt_meas_C_per_s":     "dT/dt [°C/s]",
        "dV_dt_meas_V_per_s":     "dV/dt [V/s]",
    }
    for ch, label in pretty_names.items():
        if ch not in summary:
            continue
        s = summary[ch]
        lines.append(
            f"  {label:<22} mean={s['mean']:>9.4f}  "
            f"min={s['min']:>9.4f}  max={s['max']:>9.4f}  "
            f"first={s['first']:>9.4f}  last={s['last']:>9.4f}  "
            f"trend={s['trend']}"
        )
    return "\n".join(lines)


def _format_peer(peer: Dict[str, Any], idx: int,
                 include_outcome: bool = True) -> str:
    """
    Render one verified peer-trajectory record.

    When ``include_outcome=False`` the realised outcome line is omitted,
    producing a state-only peer record used for the ablation study.
    """
    snap = peer.get("snapshot", {})
    out  = peer.get("outcome", {})
    lines = [
        f"Peer {idx} [chain_tx_hash: {peer.get('chain_tx_hash','')[:16]}…]",
        f"  scenario: {peer.get('scenario','?')}  "
        f"(family {peer.get('cause_family','?')})",
        f"  snapshot at t={peer.get('snapshot_at_s',0):.0f}s of run: "
        f"V={snap.get('voltage_V','?')}V  "
        f"T={snap.get('temperature_C','?')}°C  "
        f"dT/dt={snap.get('dT_dt_C_per_s','?')}°C/s  "
        f"SOC={snap.get('SOC','?')}",
    ]
    if include_outcome:
        lines.append(
            f"  outcome: failure_stage_max={out.get('failure_stage_max','?')}  "
            f"escalated={out.get('did_escalate', False)}  "
            f"reached_failure={out.get('reached_failure', False)}  "
            f"run_duration={out.get('duration_s',0):.0f}s"
        )
    return "\n".join(lines)


def build_prompt_v14(
    scenario:        Dict[str, Any],
    peer_records:    Optional[List[Dict[str, Any]]] = None,
    show_peers:      bool = True,
    tamper_notice:   bool = False,
    include_outcome: bool = True,
) -> str:
    """
    Build the v14 hybrid (predict + decide) prompt.

    Parameters
    ----------
    scenario        : v14 scenario dict
    peer_records    : list of peer dicts to include (overrides scenario.asset_records)
    show_peers      : if False, omit the verified-peers block (C1)
    tamper_notice   : if True, prepend a notice that some records were rejected
                      by on-chain verification and recovered from canonical store
    include_outcome : if False, peer records are presented WITHOUT their
                      realised escalation outcome.  Used by the state-only
                      ablation study to isolate the contribution of
                      outcome-bearing evidence from the contribution of
                      verification alone.
    """
    peers = peer_records if peer_records is not None \
            else scenario.get("asset_records", [])

    # Header
    md = scenario.get("metadata", {})
    sc_name = md.get("v14_scenario", "?")
    family  = md.get("cause_family_name", "?")
    t_dec   = md.get("t_decision_s", 0)
    duration = md.get("run_duration_s", 0)

    header = (
        f"=== Battery Safety Decision Task ===\n"
        f"Asset: {scenario.get('asset_id','?')}  "
        f"({sc_name}, family={family})\n"
        f"Decision time: t={t_dec:.0f}s  /  run duration so far: {duration:.0f}s\n"
    )

    # Observable summary
    obs_block = (
        f"\n=== Observable Channels (60s window ending at t_decision) ===\n"
        f"{_format_obs_window(scenario.get('obs_window', {}))}\n"
    )

    # Verified peers (RAG)
    if show_peers and peers:
        peer_lines = [
            "\n=== Verified Historical Peer Trajectories (PureChain-anchored) ===",
        ]
        if tamper_notice:
            peer_lines.append(
                "NOTE: One or more retrieved records failed on-chain verification "
                "and were excluded; canonical records were re-fetched from PureChain."
            )
        if include_outcome:
            peer_lines.append(
                "Each peer is a verified historical run with similar observable "
                "signature, plus its observed outcome.  Use peer outcomes as evidence."
            )
        else:
            peer_lines.append(
                "Each peer is a verified historical run with similar observable "
                "signature (state-only; outcomes are not disclosed)."
            )
        for i, p in enumerate(peers, 1):
            peer_lines.append(_format_peer(p, i, include_outcome=include_outcome))
        peer_block = "\n".join(peer_lines) + "\n"
    elif show_peers:
        peer_block = (
            "\n=== Verified Historical Peer Trajectories ===\n"
            "(no verified peers retrieved)\n"
        )
    else:
        peer_block = ""

    # Action menu
    actions_block = "\n=== Available Actions ===\n"
    for tok, desc in V14_ACTIONS:
        actions_block += f"  • {tok:<24}— {desc}\n"

    # Deterministic stage and prevention thresholds + action mapping.
    # Anchoring p300 to numeric thresholds prevents the LLM from
    # over-predicting Stage-2 escalation purely on the basis of
    # 'trends are rising'.
    if include_outcome:
        forecast_block = (
            "\n=== Stage-2 in 5 min Forecast (use VERIFIED PEER OUTCOMES) ===\n"
            "  Count peers whose outcome shows failure_stage_max >= 2"
            " (escalated=True).\n"
            "  YES if >= 3 of the 5 peers escalated (majority verified evidence).\n"
            "  YES OVERRIDE if dT/dt_max >= 0.30°C/s (active runaway in progress).\n"
            "  NO otherwise.\n"
            "  Rationale: similar verified batteries that escalated to Stage-2+\n"
            "  are the strongest signal for what this battery will do next.\n"
        )
        rationale = (
            "Do NOT escalate beyond what the thresholds above support. Peer\n"
            "outcomes inform the Stage-2 forecast; the thresholds set the floor.\n"
        )
    else:
        forecast_block = (
            "\n=== Stage-2 in 5 min Forecast (thresholds only) ===\n"
            "  YES if dT/dt_max >= 0.30°C/s (active runaway in progress)\n"
            "       OR temperature_max >= 70°C (committed self-heating).\n"
            "  NO otherwise.\n"
            "  Note: peer outcomes are not disclosed in this configuration;\n"
            "  rely only on the observable thresholds above.\n"
        )
        rationale = (
            "Do NOT escalate beyond what the thresholds above support.\n"
        )
    mapping_block = (
        "\n=== Stage Classification (apply STRICTLY by these thresholds) ===\n"
        "  Stage 3: temperature_max >= 110°C  OR  dT/dt_max >= 1.0°C/s\n"
        "  Stage 2: temperature_max >= 90°C   OR  dT/dt_max >= 0.30°C/s\n"
        "  Stage 1: voltage_max >= 4.25V  OR  voltage_min <= 2.30V  OR\n"
        "           temperature_max >= 45°C  OR  dT/dt_max >= 0.05°C/s\n"
        "  Stage 0: none of the above\n"
        f"{forecast_block}"
        "\n=== Action Selection Rule (apply DETERMINISTICALLY) ===\n"
        "  IF stage >= 2                          -> urgent\n"
        "  ELIF stage = 1 AND Stage-2_5min = yes  -> replace\n"
        "  ELSE                                   -> safe\n"
        f"{rationale}"
    )

    # Task instruction
    task_block = (
        "\n=== Decision Format ===\n"
        "Reasoning: <one or two sentences citing the observable channels and "
        "at least one peer outcome>\n"
        "Predicted stage: <0|1|2|3>\n"
        "Stage-2 in 5 min: <yes|no>\n"
        "Action: <derive from the rule above; exactly one of the 5 tokens>\n"
    )

    return header + obs_block + peer_block + actions_block + mapping_block + task_block


# ── v14-aware decision pipeline ───────────────────────────────

FAKE_HASH = "deadbeef" + "f" * 56


def _verify_peer_records(
    peers, pc_client,
) -> Tuple[List[Dict[str, Any]], int]:
    """C3 hash verification on peer chain_tx_hash; returns (verified, n_rejected)."""
    verified, n_rej = [], 0
    for p in peers:
        h = p.get("chain_tx_hash", "")
        if pc_client.verify_tx_hash(h):
            verified.append(p)
        else:
            n_rej += 1
    return verified, n_rej


# ── C4: content-bound verification + chain-based recovery (D1/D2) ─

_V14_TX_REGISTRY_PATH = Path("data/synthetic/v14_tx_registry.json")
_v14_tx_registry_cache: Optional[Dict[str, Any]] = None


def _v14_tx_registry() -> Dict[str, Any]:
    """Identity -> {tx_hash, digest} index built by the anchoring
    campaign (scripts/anchor_v14_peer_digests.py)."""
    global _v14_tx_registry_cache
    if _v14_tx_registry_cache is None:
        try:
            _v14_tx_registry_cache = json.loads(
                _V14_TX_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            _v14_tx_registry_cache = {}
    return _v14_tx_registry_cache


def _verify_peer_records_content(
    peers, pc_client,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """C4 content-bound verification: the handle must resolve to a
    committed v14 peer event AND the on-chain content digest must match
    the digest of the locally retrieved record.  Returns
    (verified, rejected)."""
    verified, rejected = [], []
    for p in peers:
        h = p.get("chain_tx_hash", "")
        d = peer_content_digest(p)
        if h and pc_client.verify_content(h, d):
            verified.append(p)
        else:
            rejected.append(p)
    return verified, rejected


def _recover_peer_from_chain(
    peer, pc_client,
) -> Optional[Dict[str, Any]]:
    """Genuine chain-based recovery of a rejected peer record.

    Candidate handles: the peer's own (possibly tampered) handle, then
    the identity registry's handle for this peer.  A candidate is
    accepted only if (i) its committed payload hashes to the digest
    committed in the same event and (ii) the payload's identity fields
    match the requested peer, so a replayed or mis-mapped handle cannot
    substitute a different genuine record.
    """
    key = peer_key(peer)
    candidates = [peer.get("chain_tx_hash", "")]
    reg = _v14_tx_registry().get(key)
    if reg:
        candidates.append(reg.get("tx_hash", ""))
    for h in candidates:
        if not h:
            continue
        got = pc_client.get_peer_payload(h)
        if got is None:
            continue
        payload_json, digest = got
        if hashlib.sha256(payload_json.encode()).hexdigest() != digest:
            continue  # committed payload/digest inconsistent
        try:
            rec = json.loads(payload_json)
        except Exception:
            continue
        if peer_key(rec) != key:
            continue  # identity mismatch (replayed handle)
        rec["chain_tx_hash"] = h.replace("0x", "")
        rec["content_digest"] = digest
        rec["_is_v14_peer"] = True
        rec["_recovered_from_chain"] = True
        return rec
    return None


def decide_v14(
    agent,                    # LLMAgent (avoid circular import in type hint)
    scenario:        Dict[str, Any],
    condition:       str = "C3",
    canonical_peers: Optional[List[Dict[str, Any]]] = None,
    include_outcome: bool = True,
) -> Dict[str, Any]:
    """
    v14 decision pipeline.

      C0: pure rule engine on observable summary
      C1: LLM with observables only (no peers)
      C2: LLM with peers as-is (no hash check)
      C3: LLM with on-chain-verified peers; on tamper detection, fall
          back to canonical_peers (PureChain-recovered set).

    ``include_outcome=False`` runs the state-only ablation in which the
    LLM sees verified peers but without their realised escalation
    outcomes; the Stage-2 forecast rule then falls back to thresholds.
    """
    if condition == "C0":
        result = c0_decide_v14(scenario)
        return result

    t_start = time.perf_counter()
    peers   = scenario.get("asset_records", []) or []
    tamper_alerts  = 0
    peers_recovered = 0
    peers_for_llm: List[Dict[str, Any]] = []
    show_peers     = True
    tamper_notice  = False

    if condition == "C1":
        show_peers = False
    elif condition == "C2":
        peers_for_llm = peers
    elif condition == "C3":
        v_peers, n_rej = _verify_peer_records(peers, agent.pc_client)
        tamper_alerts = n_rej
        if n_rej > 0 and canonical_peers is not None:
            peers_for_llm = canonical_peers
            tamper_notice = True
        else:
            peers_for_llm = v_peers
    elif condition == "C4":
        # Content-bound verification + genuine chain-based recovery.
        # No oracle: rejected peers are reconstructed from committed
        # PureChain payloads; the observable header stays as reported.
        v_peers, rejected = _verify_peer_records_content(peers, agent.pc_client)
        tamper_alerts = len(rejected)
        recovered: List[Dict[str, Any]] = []
        for p in rejected:
            r = _recover_peer_from_chain(p, agent.pc_client)
            if r is not None:
                recovered.append(r)
        peers_recovered = len(recovered)
        peers_for_llm = v_peers + recovered
        tamper_notice = tamper_alerts > 0

    prompt = build_prompt_v14(
        scenario,
        peer_records=peers_for_llm if condition != "C1" else None,
        show_peers=show_peers,
        tamper_notice=tamper_notice,
        include_outcome=include_outcome,
    )

    t_inf = time.perf_counter()
    try:
        if agent._is_anthropic_model():
            llm_output = agent._call_anthropic(prompt)
        elif agent._is_openai_model():
            llm_output = agent._call_openai(prompt)
        else:
            llm_output = agent._call_ollama(prompt)
    except Exception as exc:
        logger.error(f"[v14 {condition}] LLM call failed: {exc}")
        llm_output = ""
    latency_inference_ms = (time.perf_counter() - t_inf) * 1000

    parsed = parse_v14_output(llm_output)
    action = parsed["action"] or "uncertain"
    classification = "valid" if action in V14_ACTION_TOKENS else "uncertain"

    latency_total = (time.perf_counter() - t_start) * 1000

    return {
        "decision":             action,
        "predicted_stage":      parsed["predicted_stage"],
        "predicted_p300":       parsed["predicted_p300"],
        "classification":       classification,
        "trace_rate":           1.0 if condition in ("C2", "C3", "C4") and peers_for_llm else 0.0,
        "latency_ms":           round(latency_total, 1),
        "latency_retrieval_ms": 0.0,
        "latency_inference_ms": round(latency_inference_ms, 1),
        "model_used":           agent.model,
        "tamper_alerts":        tamper_alerts,
        "peers_recovered":      peers_recovered,
        "condition":            condition,
        "llm_output":           llm_output,
        "violations":           [],
    }
