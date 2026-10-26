"""Unit tests for the v14 physics-consistency gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.layer3.v14_physics_checker import (  # noqa: E402
    check_v14, apply_fallback, stage_bounds,
    POLICY_CLAMP, POLICY_DEFER, POLICY_CLAMP_THEN_DEFER, POLICY_OFF,
)
from src.layer1.v14_rule_engine import _extract_features  # noqa: E402


def _obs(v_max=3.7, v_min=3.6, t_max=25.0, t_last=25.0,
         dtdt_max=0.001, dtdt_last=0.001, soc=0.8, i_max=1.0):
    """Build an observable summary in the shape the extractor expects."""
    def ch(mean, mn, mx, first, last):
        return {"mean": mean, "min": mn, "max": mx,
                "first": first, "last": last, "trend": "flat"}
    return {
        "voltage_V_measured":     ch(v_max, v_min, v_max, v_min, v_max),
        "temperature_C_measured": ch(t_max, 20.0, t_max, 20.0, t_last),
        "dT_dt_meas_C_per_s":     ch(dtdt_max, 0.0, dtdt_max, 0.0, dtdt_last),
        "current_A_measured":     ch(i_max, 0.0, i_max, 0.0, i_max),
        "SOC_with_drift":         ch(soc, soc, soc, soc, soc),
    }


QUIET   = _obs()                                   # nothing happening
ABUSE   = _obs(v_max=4.40, t_max=48.0, t_last=47.0, dtdt_max=0.06)
CASCADE = _obs(t_max=95.0, t_last=94.0, dtdt_max=0.45, dtdt_last=0.45)
RUNAWAY = _obs(t_max=125.0, t_last=124.0, dtdt_max=1.40, dtdt_last=1.40)


def test_bounds_ladder():
    assert stage_bounds(_extract_features(QUIET))   == (0, 1)
    assert stage_bounds(_extract_features(ABUSE))   == (1, 2)
    assert stage_bounds(_extract_features(CASCADE)) == (2, 3)
    assert stage_bounds(_extract_features(RUNAWAY)) == (3, 3)


def test_valid_outputs_pass():
    assert check_v14(0, 0, "safe", QUIET).ok
    assert check_v14(1, 1, "replace", ABUSE).ok
    assert check_v14(2, 1, "urgent", CASCADE).ok
    assert check_v14(3, 1, "urgent", RUNAWAY).ok


def test_reviewer_case_consistent_but_physically_wrong():
    """R2.6: a mutually consistent triple that is physically impossible.

    Quiet 60 s window, but the model claims a runaway in progress.
    stage>=2 maps to 'urgent', so the triple is internally consistent --
    an internal-consistency-only checker would pass it.
    """
    r = check_v14(3, 1, "urgent", QUIET)
    assert not r.ok
    assert "stage_overstated" in r.violations
    assert "action_mapping_violation" not in r.violations


def test_under_triage_is_caught():
    """The safety-critical direction: runaway observables called safe."""
    r = check_v14(0, 0, "safe", RUNAWAY)
    assert not r.ok
    assert "stage_understated" in r.violations


def test_action_mapping_violation():
    r = check_v14(2, 1, "safe", CASCADE)
    assert not r.ok
    assert "action_mapping_violation" in r.violations


def test_p300_definitional_inconsistency():
    r = check_v14(2, 0, "urgent", CASCADE)
    assert not r.ok
    assert "p300_inconsistent_with_stage" in r.violations


def test_unparsable_action():
    r = check_v14(1, 0, None, ABUSE)
    assert not r.ok
    assert "action_unparsable" in r.violations


def test_fallback_clamp_projects_into_band():
    r = check_v14(3, 1, "urgent", ABUSE)          # stage 3 unsupported (hi=2)
    assert "stage_overstated" in r.violations
    out = apply_fallback(r, 3, 1, "urgent", ABUSE, policy=POLICY_CLAMP)
    assert out["predicted_stage"] == 2            # clamped to the upper bound
    assert out["decision"] == "urgent"
    assert out["fallback_applied"] == "clamp"


def test_fallback_defer_uses_rule_engine():
    r = check_v14(3, 1, "urgent", QUIET)
    out = apply_fallback(r, 3, 1, "urgent", QUIET, policy=POLICY_DEFER)
    assert out["decision"] == "safe"
    assert out["predicted_stage"] == 0
    assert out["fallback_applied"] == "defer"


def test_clamp_then_defer_escalates_to_defer():
    """Clamping alone leaves an unsupported p300, so the gate defers."""
    r = check_v14(3, 1, "urgent", QUIET)
    out = apply_fallback(r, 3, 1, "urgent", QUIET,
                         policy=POLICY_CLAMP_THEN_DEFER)
    assert out["fallback_applied"] == "defer"
    assert out["decision"] == "safe"


def test_policy_off_is_passthrough():
    r = check_v14(3, 1, "urgent", QUIET)
    out = apply_fallback(r, 3, 1, "urgent", QUIET, policy=POLICY_OFF)
    assert out["decision"] == "urgent"
    assert out["fallback_applied"] == "none"


def test_valid_output_is_never_modified():
    r = check_v14(2, 1, "urgent", CASCADE)
    out = apply_fallback(r, 2, 1, "urgent", CASCADE,
                         policy=POLICY_CLAMP_THEN_DEFER)
    assert out["fallback_applied"] == "none"
    assert out["decision"] == "urgent"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("  PASS  %s" % fn.__name__)
        except Exception:
            failed += 1
            print("  FAIL  %s" % fn.__name__)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
