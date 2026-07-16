"""
retry.py
Layer 3 — Two-stage retry, Stage 2 (Algorithm 3).

When Stage-1 GPT-4o fails the physics check (invalid or uncertain),
stage2_retry() is called. It rebuilds the prompt with the failure
reason appended, calls Llama 3 70B via Ollama, and re-runs the
physics checker on the response.

If Ollama is unreachable the function returns None so the caller
can handle gracefully (mark as uncertain, log anyway).
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.logger_setup import setup_logger
from src.config import LLM_SECONDARY, LLM_TEMPERATURE, SYSTEM_PROMPT, OLLAMA_BASE_URL
from src.layer3.physics_checker import PhysicsChecker, CheckResult

setup_logger("llm_agent")   # shared log file with llm_agent.py
load_dotenv()

_checker = PhysicsChecker()


# ── Stage-2 entry point ────────────────────────────────────────

def stage2_retry(
    scenario:       Dict[str, Any],
    rag_context:    Dict[str, Any],
    failure_reason: str,
) -> Optional[Dict[str, Any]]:
    """
    Algorithm 3, Stage 2 — Llama 3 70B via Ollama.

    Appends the physics checker failure reason to the original prompt
    and asks Llama 3 70B to try again.

    Parameters
    ----------
    scenario       : full scenario dict from scenarios.json
    rag_context    : retrieval result dict (asset_records,
                     routing_records, prompt_context, tamper_alerts)
    failure_reason : human-readable description of why Stage-1 failed

    Returns
    -------
    dict with keys: llm_output, check (CheckResult), model_used, latency_ms
    or None if Ollama is not reachable.
    """
    # Late import to avoid circular dependency at module load
    from src.layer3.llm_agent import _build_prompt

    scenario_id = scenario.get("scenario_id", "unknown")
    deg         = scenario.get("degradation_pct")

    # Rebuild the original prompt (same as Stage-1) and append failure note
    base_prompt   = _build_prompt(scenario, rag_context.get("prompt_context", ""),
                                  rag_dict=rag_context if rag_context.get("asset_records") else None)
    retry_prompt  = (
        f"{base_prompt}\n\n"
        "=== Physics Check Failure (Stage-1) ===\n"
        f"Reason: {failure_reason}\n\n"
        "Please re-evaluate using ONLY the retrieved supply records above. "
        "State the exact SOC, temperature, and degradation values before "
        "giving your recommendation. Cite the chain_tx_hash of every record "
        "you use."
    )

    ollama_url = OLLAMA_BASE_URL
    model      = LLM_SECONDARY   # gemma3:4b (or whatever LLM_SECONDARY is set to)

    logger.debug(
        f"Stage-2 Ollama call: model={model} url={ollama_url} "
        f"scenario={scenario_id[:8]}..."
    )

    t0 = time.perf_counter()
    llm_output = _call_ollama(retry_prompt, model, ollama_url)
    latency_ms = (time.perf_counter() - t0) * 1000

    if llm_output is None:
        return None

    logger.info(
        f"Stage-2 {model} inference={latency_ms:.0f}ms  "
        f"scenario={scenario_id[:8]}..."
    )
    logger.debug(f"Stage-2 output:\n{llm_output[:500]}")

    check = _checker.check(llm_output, rag_context, degradation_pct=deg)
    logger.info(
        f"Stage-2 [{model}] physics=[{check.classification}]  "
        f"tau={check.trace_rate:.2f}  action={check.llm_action}"
    )

    return {
        "llm_output": llm_output,
        "check":      check,
        "model_used": model,
        "latency_ms": round(latency_ms, 1),
    }


# ── Ollama HTTP call ───────────────────────────────────────────

def _call_ollama(
    prompt:      str,
    model:       str,
    base_url:    str,
) -> Optional[str]:
    """
    POST to Ollama /api/chat. Returns the assistant content string,
    or None if the server is unreachable or returns an error.
    """
    import json
    try:
        import urllib.request
        payload = json.dumps({
            "model":    model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream":  False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": 200,   # cap tokens to reduce inference latency
            },
        }).encode()

        req = urllib.request.Request(
            url=f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            return body["message"]["content"]

    except Exception as exc:
        logger.warning(
            f"Ollama Stage-2 call failed: {exc}  "
            f"(Is Ollama running? start with: ollama serve)"
        )
        return None
