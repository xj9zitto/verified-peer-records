"""
v14_rule_engine.py — physics-threshold rule engine for C0 baseline on the
v14 battery safety task.

Predicts failure_stage and prevention_label_300s from the observable
channels in scenario["obs_window"]["summary"].  Then maps the prediction
to an action from the 5-item menu using the same deterministic table the
ground-truth uses.

Thresholds are anchored to the v14 dataset's own scenario design (see
SCENARIO_TRACEABILITY.md) and to standard battery-safety reference values:

  Voltage:
    >  4.25V  → overcharge onset (Liu 2018)
    <  2.30V  → Cu-dissolution onset (Guo 2016)

  Temperature:
    >  45°C   → thermal stress
    >  60°C   → SEI onset proximate
    >  90°C   → separator integrity at risk
    > 110°C   → IEC 62133-2 oven-test failure threshold

  Rate of temperature rise:
    >  0.05°C/s  → mild self-heating
    >  0.30°C/s  → cascade onset
    >  1.00°C/s  → runaway in progress
"""

from __future__ import annotations
import time
from typing import Any, Dict, Tuple


# ── Action mapping (mirrors v14_loader._gt_action) ────────────

def gt_action_from(stage: int, p300: int, p3600: int) -> str:
    """3-class action mapping (matches v14_loader._gt_action)."""
    if stage >= 2:
        return "urgent"
    if stage == 1 and p300 == 1:
        return "replace"
    return "safe"


# ── Threshold constants ───────────────────────────────────────

V_OVERCHARGE     = 4.25
V_OVERDISCHARGE  = 2.30

T_WARN           = 45.0
T_CRITICAL       = 60.0
T_SEPARATOR      = 90.0
T_RUNAWAY        = 110.0

DTDT_WARN        = 0.05
DTDT_CRITICAL    = 0.30
DTDT_RUNAWAY     = 1.00

# Prevention horizon heuristic: if current state is at warning AND any
# rate signal is in the critical band, future Stage-2 within 5min is likely.
DTDT_FUTURE_GATE = 0.10
T_MARGIN_GATE    = 8.0          # within 8°C of T_CRITICAL


def _extract_features(obs_summary: Dict[str, Any]) -> Dict[str, float]:
    """Pull the channels the rule engine cares about; default to 0.0 if missing."""
    def get(channel: str, stat: str) -> float:
        d = obs_summary.get(channel, {})
        v = d.get(stat)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    return {
        "v_max":      get("voltage_V_measured",      "max"),
        "v_min":      get("voltage_V_measured",      "min"),
        "v_last":     get("voltage_V_measured",      "last"),
        "t_max":      get("temperature_C_measured",  "max"),
        "t_last":     get("temperature_C_measured",  "last"),
        "dtdt_max":   get("dT_dt_meas_C_per_s",      "max"),
        "dtdt_last":  get("dT_dt_meas_C_per_s",      "last"),
        "i_max":      get("current_A_measured",      "max"),
        "soc_last":   get("SOC_with_drift",          "last"),
    }


def predict_stage(features: Dict[str, float]) -> int:
    """Map physics features to {0,1,2,3} predicted failure_stage."""
    # Stage 3 — runaway in progress
    if features["t_max"] >= T_RUNAWAY:
        return 3
    if features["dtdt_max"] >= DTDT_RUNAWAY:
        return 3

    # Stage 2 — separator at risk OR self-heating committed
    if features["t_max"] >= T_SEPARATOR:
        return 2
    if features["dtdt_max"] >= DTDT_CRITICAL:
        return 2
    if features["t_max"] >= T_CRITICAL and features["dtdt_max"] >= DTDT_WARN:
        return 2

    # Stage 1 — abuse signature without committed cascade
    if features["v_max"] >= V_OVERCHARGE:
        return 1
    if features["v_min"] <= V_OVERDISCHARGE:
        return 1
    if features["t_max"] >= T_WARN:
        return 1
    if features["dtdt_max"] >= DTDT_WARN:
        return 1

    return 0


def predict_prevention_300s(features: Dict[str, float], stage: int) -> int:
    """
    Will Stage>=2 occur in (t, t+300s]?  Heuristic:
      - if current stage is already 2/3, return 1 trivially
      - if stage 1 and rate signals say 'climbing fast', return 1
      - else 0
    """
    if stage >= 2:
        return 1
    if stage == 1 and features["dtdt_max"] >= DTDT_FUTURE_GATE:
        return 1
    if stage == 1 and features["t_last"] >= (T_CRITICAL - T_MARGIN_GATE):
        return 1
    return 0


def predict_prevention_3600s(features: Dict[str, float], stage: int, p300: int) -> int:
    """
    1h window — only flag if we have *real* evidence of escalation, not just
    any rate signal.  Setting this too eagerly misclassifies stable
    stage-0 scenarios as 'monitor' and was the dominant C0 error mode.
    """
    if p300 == 1:
        return 1
    if stage >= 1:
        return 1
    return 0


# ── Public C0 entry point ─────────────────────────────────────

def c0_decide_v14(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure rule-engine decision over the v14 observable summary.
    Returns the same record schema as conditions._c0_decide() / agent.decide().
    """
    t0 = time.perf_counter()
    obs_summary = scenario.get("obs_window", {}).get("summary", {})
    features    = _extract_features(obs_summary)

    stage = predict_stage(features)
    p300  = predict_prevention_300s(features, stage)
    p3600 = predict_prevention_3600s(features, stage, p300)
    action = gt_action_from(stage, p300, p3600)

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "decision":             action,
        "predicted_stage":      stage,
        "predicted_p300":       p300,
        "predicted_p3600":      p3600,
        "classification":       "valid",
        "trace_rate":           0.0,
        "latency_ms":           round(latency_ms, 2),
        "latency_retrieval_ms": 0.0,
        "latency_inference_ms": 0.0,
        "model_used":           "rule_engine_v14",
        "tamper_alerts":        0,
        "condition":            "C0",
        "llm_output":           "",
        "violations":           [],
    }
