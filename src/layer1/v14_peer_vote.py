"""
v14_peer_vote.py — a deterministic, non-LLM decision rule that consumes
the same verified peer records as the LLM agent.

Purpose: isolate the contribution of the verification mechanism from the
contribution of the LLM/RAG architecture.  If content-bound verification
protects this classifier under adversarial input in the same way it
protects the LLM agent, the robustness is a property of the verification
layer rather than of the specific reasoning engine.

The rule mirrors the decision guidance given to the LLM in the prompt:

  stage_peer  modal realised failure stage across the retrieved peers
              (ties resolved upward, i.e. conservatively)
  p300_peer   1 if at least half the peers escalated
  stage_obs   the stage the observable window compels (rule engine)

  stage_hat = max(stage_obs, stage_peer)

The observable floor keeps the classifier sensitive to falsified
telemetry, exactly as the LLM is, so both architectures face the same
attack surface.  The action follows the same deterministic map used for
the ground truth.
"""

from __future__ import annotations
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.layer1.v14_rule_engine import (
    _extract_features, gt_action_from,
    predict_stage, predict_prevention_300s,
)


def _peer_evidence(peers: List[Dict[str, Any]]) -> Tuple[int, int]:
    """(modal peer stage, majority-escalation indicator)."""
    stages, escal = [], []
    for p in peers:
        out = p.get("outcome") or {}
        try:
            stages.append(int(out.get("failure_stage_max", 0)))
        except (TypeError, ValueError):
            continue
        escal.append(bool(out.get("did_escalate", False)))

    if not stages:
        return 0, 0

    cnt = Counter(stages)
    top = max(cnt.values())
    stage_peer = max(s for s, c in cnt.items() if c == top)
    need = math.ceil(len(escal) / 2)
    p300_peer = 1 if sum(escal) >= need else 0
    return stage_peer, p300_peer


def evidence_vote(
    scenario: Dict[str, Any],
    peers:    Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Decide from the observable window plus (optionally) a set of peer
    records.  Passing ``peers=None`` or an empty list yields an
    observables-only decision.
    """
    obs = (scenario.get("obs_window") or {}).get("summary", {})
    f = _extract_features(obs)

    stage_obs = predict_stage(f)
    stage_peer, p300_peer = _peer_evidence(peers or [])

    stage_hat = max(stage_obs, stage_peer)
    p300_hat = max(p300_peer, predict_prevention_300s(f, stage_hat))

    return {
        "decision":        gt_action_from(stage_hat, p300_hat, 0),
        "predicted_stage": stage_hat,
        "predicted_p300":  p300_hat,
        "stage_obs":       stage_obs,
        "stage_peer":      stage_peer,
        "n_peers":         len(peers or []),
    }
