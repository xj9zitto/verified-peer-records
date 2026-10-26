"""
v14_physics_checker.py — physics-consistency gate for the v14 battery
safety task (Algorithm 1, ``check`` step).

The checker enforces two independent requirements on a model output
(predicted stage, predicted Stage-2 indicator, selected action):

  1. **Internal consistency.**  The action must equal the deterministic
     mapping of the predicted stage and prevention indicator, i.e. the
     same table the ground truth uses (``gt_action_from``).  This alone
     is insufficient: a mutually consistent triple can still be
     physically wrong.

  2. **Independent physical admissibility.**  The predicted stage must
     lie inside the interval of stages the observable window can
     actually support.  The interval is bounded below by the stage the
     evidence *compels* and above by the stage the evidence *permits*.

The admissible interval, rather than a point prediction, is deliberate.
Checking the model against a point estimate would make the gate
equivalent to the rule-engine baseline and would therefore be circular:
the agent would be scored against the very baseline it is compared with.
An interval encodes only what the physics rules out, leaving genuine
inferential freedom (for example, a latent internal short that has not
yet produced a thermal signature) inside the band.

Bounds are anchored to the same reference values as the rule engine
(see ``src/layer1/v14_rule_engine.py``):

  lower bound (evidence compels at least this stage)
    T_max >= 110 C  or  dT/dt_max >= 1.00 C/s   -> stage 3
    T_max >=  90 C  or  dT/dt_max >= 0.30 C/s   -> stage 2
    V_max >= 4.25 V, V_min <= 2.30 V,
    T_max >=  45 C, or dT/dt_max >= 0.05 C/s    -> stage 1
    otherwise                                    -> stage 0

  upper bound (evidence permits at most this stage)
    lower bound >= 2                             -> 3
    any stage-1 abuse signature present          -> 2
    quiet observables                            -> 1

The upper bound encodes the physical fact that Stage >= 2 is *defined*
by an active thermal cascade, which is directly observable in T and
dT/dt.  A quiet 60 s window therefore cannot support a Stage-2 or
Stage-3 claim, whereas it can still support Stage 1 (an abuse condition
may be present without a thermal signature yet).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.layer1.v14_rule_engine import (
    _extract_features,
    gt_action_from,
    predict_stage,
    predict_prevention_300s,
    V_OVERCHARGE, V_OVERDISCHARGE,
    T_WARN, T_CRITICAL, T_SEPARATOR, T_RUNAWAY,
    DTDT_WARN, DTDT_CRITICAL, DTDT_RUNAWAY,
    DTDT_FUTURE_GATE, T_MARGIN_GATE,
)

# Fallback policies
POLICY_OFF              = "off"
POLICY_CLAMP            = "clamp"
POLICY_DEFER            = "defer"
POLICY_CLAMP_THEN_DEFER = "clamp_then_defer"

VALID_POLICIES = {POLICY_OFF, POLICY_CLAMP, POLICY_DEFER, POLICY_CLAMP_THEN_DEFER}


@dataclass
class CheckResult:
    ok:          bool
    violations:  List[str] = field(default_factory=list)
    stage_lo:    int = 0
    stage_hi:    int = 3

    def as_list(self) -> List[str]:
        return list(self.violations)


# ── Admissible stage interval ─────────────────────────────────

def _has_abuse_signature(f: Dict[str, float]) -> bool:
    """Any observable signature of an abuse condition (stage >= 1)."""
    return (f["v_max"]    >= V_OVERCHARGE
            or f["v_min"] <= V_OVERDISCHARGE
            or f["t_max"] >= T_WARN
            or f["dtdt_max"] >= DTDT_WARN)


def stage_bounds(features: Dict[str, float]) -> Tuple[int, int]:
    """
    Return (lo, hi): the minimum stage the observables compel and the
    maximum stage they can support.
    """
    # Lower bound: the rule ladder states what the evidence forces.
    lo = predict_stage(features)

    # Upper bound: what the evidence can support.
    if lo >= 2:
        hi = 3
    elif _has_abuse_signature(features):
        # Abuse present but no committed thermal cascade: a transition
        # into stage 2 is arguable, stage 3 is not.
        hi = 2
    else:
        # Quiet window: stage >= 2 requires an observable thermal
        # cascade, so it cannot be supported.  Stage 1 remains possible
        # (e.g. a latent internal short before thermal onset).
        hi = 1

    return lo, hi


# ── The gate ──────────────────────────────────────────────────

def check_v14(
    predicted_stage: Any,
    predicted_p300:  Any,
    action:          Any,
    obs_summary:     Dict[str, Any],
) -> CheckResult:
    """
    Physics-consistency gate.  Returns a CheckResult whose ``violations``
    list is empty when the output is both internally consistent and
    physically admissible.
    """
    violations: List[str] = []
    f = _extract_features(obs_summary or {})
    lo, hi = stage_bounds(f)

    # --- Well-formedness -------------------------------------
    if action not in {"safe", "replace", "urgent"}:
        violations.append("action_unparsable")
    if predicted_stage is None:
        violations.append("stage_missing")
    if predicted_p300 is None:
        violations.append("p300_missing")

    # Without a stage we cannot run the physical checks.
    if predicted_stage is None:
        return CheckResult(ok=not violations, violations=violations,
                           stage_lo=lo, stage_hi=hi)

    try:
        stage = int(predicted_stage)
    except (TypeError, ValueError):
        violations.append("stage_malformed")
        return CheckResult(ok=False, violations=violations,
                           stage_lo=lo, stage_hi=hi)

    if stage < 0 or stage > 3:
        violations.append("stage_out_of_range")

    # --- 2. Independent physical admissibility ---------------
    # Understating is the safety-critical direction (under-triage).
    if stage < lo:
        violations.append("stage_understated")
    if stage > hi:
        violations.append("stage_overstated")

    # --- 1. Internal consistency -----------------------------
    if predicted_p300 is not None:
        try:
            p300 = int(predicted_p300)
        except (TypeError, ValueError):
            p300 = None
            violations.append("p300_malformed")

        if p300 is not None:
            # Stage >= 2 means Stage 2 has already been reached, so the
            # "Stage 2 within 300 s" indicator is true by definition.
            if stage >= 2 and p300 != 1:
                violations.append("p300_inconsistent_with_stage")

            # A quiet window with no rate signal cannot support an
            # imminent-escalation claim.
            if (hi == 1 and p300 == 1
                    and f["dtdt_max"] < DTDT_FUTURE_GATE
                    and f["t_last"] < (T_CRITICAL - T_MARGIN_GATE)):
                violations.append("p300_unsupported")

            if action in {"safe", "replace", "urgent"}:
                expected = gt_action_from(stage, p300, 0)
                if action != expected:
                    violations.append("action_mapping_violation")

    return CheckResult(ok=not violations, violations=violations,
                       stage_lo=lo, stage_hi=hi)


# ── Fallback ──────────────────────────────────────────────────

def apply_fallback(
    check:           CheckResult,
    predicted_stage: Any,
    predicted_p300:  Any,
    action:          Any,
    obs_summary:     Dict[str, Any],
    policy:          str = POLICY_CLAMP,
) -> Dict[str, Any]:
    """
    Repair an output that failed the gate.

    Policies
    --------
    ``off``              no repair; the output is passed through and the
                         violations are recorded for reporting only.
    ``clamp``            project the predicted stage into the admissible
                         interval, restore the definitional p300, and
                         recompute the action from the deterministic map.
    ``defer``            discard the model output and fall back to the
                         rule engine, which is by construction admissible.
    ``clamp_then_defer`` clamp first; defer only if the clamped output is
                         still not admissible (e.g. the action could not
                         be parsed at all).

    ``clamp`` is the default because it minimises under-triage, the
    safety-critical error.  Replaying every policy over one set of model
    outputs (scripts/analyze_checker_policies.py) gives under-triage
    0.034 for clamp against 0.082 for clamp_then_defer, 0.088 for defer,
    and 0.114 ungated, and clamp is the only repairing policy that keeps
    the "replace" class usable: deferring to the rule engine inherits its
    near-zero recall there, so clamp_then_defer eliminates the class the
    task exists to serve.

    Returns a dict with the repaired fields plus ``fallback_applied``.
    """
    if policy not in VALID_POLICIES:
        raise ValueError("unknown fallback policy: %r" % (policy,))

    out = {
        "decision":         action,
        "predicted_stage":  predicted_stage,
        "predicted_p300":   predicted_p300,
        "fallback_applied": "none",
    }
    if policy == POLICY_OFF or check.ok:
        return out

    f = _extract_features(obs_summary or {})

    def _defer() -> Dict[str, Any]:
        stage = predict_stage(f)
        p300  = predict_prevention_300s(f, stage)
        return {
            "decision":         gt_action_from(stage, p300, 0),
            "predicted_stage":  stage,
            "predicted_p300":   p300,
            "fallback_applied": "defer",
        }

    if policy == POLICY_DEFER:
        return _defer()

    # clamp / clamp_then_defer
    try:
        stage = int(predicted_stage)
    except (TypeError, ValueError):
        return _defer() if policy == POLICY_CLAMP_THEN_DEFER else out

    stage = max(check.stage_lo, min(check.stage_hi, stage))

    try:
        p300 = int(predicted_p300)
    except (TypeError, ValueError):
        p300 = predict_prevention_300s(f, stage)
    if stage >= 2:
        p300 = 1

    repaired = {
        "decision":         gt_action_from(stage, p300, 0),
        "predicted_stage":  stage,
        "predicted_p300":   p300,
        "fallback_applied": "clamp",
    }

    if policy == POLICY_CLAMP_THEN_DEFER:
        recheck = check_v14(repaired["predicted_stage"],
                            repaired["predicted_p300"],
                            repaired["decision"], obs_summary)
        if not recheck.ok:
            return _defer()

    return repaired
