"""
Parity between the dependency-free attack module and the experiment
runner it was factored out of.

The combined observable-plus-store attack used to live inside
scripts/run_v14_poisoning_experiment.py, which imports the LLM agent and
transitively ChromaDB.  It now lives in src/dataset/v14_attack.py so the
ML baselines, transfer arms, recovery baseline, and boundary attacks can
reproduce the identical attack without that stack.  Identical is the
whole point: every arm must face the same attack instances for the
paired comparisons to mean anything, so this test pins the two
implementations together rather than trusting they stay in step.

Run: .\.venv\Scripts\python.exe src/tests/test_v14_attack_parity.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.v14_attack import (  # noqa: E402
    DEFAULT_RATES, FAKE_HASH, is_poisoned, attack_type, poison_scenario,
)

SCEN = Path("data/synthetic/v14_scenarios.json")
_scenarios = json.loads(SCEN.read_text()) if SCEN.exists() else []


def _runner():
    """The runner's own implementations, or None if its import stack is
    unavailable here (ChromaDB's embedding backend is optional)."""
    try:
        from scripts.run_v14_poisoning_experiment import (
            _is_poisoned, _attack_type, _poison_scenario,
        )
        return _is_poisoned, _attack_type, _poison_scenario
    except Exception as exc:
        print("     (runner import unavailable: %s)" % type(exc).__name__)
        return None


def test_draw_parity():
    """Poison set and attack assignment agree for every scenario."""
    r = _runner()
    if r is None or not _scenarios:
        return
    _is_poisoned, _attack_type, _ = r
    for sc in _scenarios:
        sid = sc["scenario_id"]
        for rate in DEFAULT_RATES:
            assert is_poisoned(sid, rate) == _is_poisoned(sid, rate), sid
        for alpha in (0.0, 0.5, 1.0):
            assert attack_type(sid, alpha) == _attack_type(sid, alpha), sid


def test_poisoned_scenario_parity():
    """The poisoned scenario is byte-identical under both attack types."""
    r = _runner()
    if r is None or not _scenarios:
        return
    _, _, _poison_scenario = r
    for sc in _scenarios:
        for atype in ("fake_hash", "forged_hash"):
            a = json.dumps(poison_scenario(sc, atype), sort_keys=True)
            b = json.dumps(_poison_scenario(sc, atype), sort_keys=True)
            assert a == b, "%s / %s" % (sc["scenario_id"], atype)


def test_attack_actually_changes_the_record():
    """Guard against a no-op attack silently passing the parity tests."""
    assert _scenarios, "scenario corpus required"
    sc = next(s for s in _scenarios
              if s.get("asset_records") and any(
                  (p.get("outcome") or {}).get("did_escalate")
                  for p in s["asset_records"]))
    p = poison_scenario(sc, "fake_hash")
    assert p["_poisoned"] is True
    assert all(not q["outcome"]["did_escalate"] for q in p["asset_records"])
    assert all(q["chain_tx_hash"] == FAKE_HASH for q in p["asset_records"])
    assert p["obs_window"]["summary"]["temperature_C_measured"]["max"] == 25.0
    # The original must be untouched: the attack copies.
    assert any((q.get("outcome") or {}).get("did_escalate")
               for q in sc["asset_records"])


def test_forged_hash_keeps_genuine_handle():
    assert _scenarios, "scenario corpus required"
    sc = next(s for s in _scenarios if s.get("asset_records"))
    p = poison_scenario(sc, "forged_hash")
    for before, after in zip(sc["asset_records"], p["asset_records"]):
        assert after["chain_tx_hash"] == before["chain_tx_hash"]


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
