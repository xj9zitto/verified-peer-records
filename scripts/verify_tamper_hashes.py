"""
verify_tamper_hashes.py
On-chain verification audit for the tamper robustness experiment.

Purpose
-------
The tamper experiment injects FAKE_HASH (not on-chain) into tampered records.
This script proves via live PureChain calls that:
  1. verify_tx_hash(FAKE_HASH)  → False  (adversarial hash rejected)
  2. verify_tx_hash(real_hash)  → True   (legitimate hash accepted)

It then updates tamper_alerts in tamper_results.json by re-running
verify_tx_hash() for every candidate record hash in each C3 scenario,
without re-running any LLM calls.  The tamper_alerts counts already
stored in the file were set by construction (n_tampered); this script
replaces them with counts derived from actual blockchain verification.

Usage:
    python scripts/verify_tamper_hashes.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from src.logger_setup import setup_logger
from src.layer2.purechain_client import PureChainClient
from src.layer2.serialiser import RecordSerialiser

setup_logger("verify_tamper")

RESULTS_DIR = Path("results")
FAKE_HASH   = "deadbeef" + "f" * 56


def main() -> None:
    client     = PureChainClient()
    serialiser = RecordSerialiser()

    print("Connecting to PureChain...")
    ok = client.connect()
    if not ok:
        print("ERROR: Could not connect to PureChain. Check PURECHAIN_RPC_URL.")
        sys.exit(1)
    print(f"  Connected: {client.is_connected}\n")

    # ── Step 1: Prove the two key verification cases ────────────
    print("=== PureChain Hash Verification Proof ===\n")

    fake_result = client.verify_tx_hash(FAKE_HASH)
    print(f"  verify_tx_hash(FAKE_HASH)  → {fake_result}")
    assert fake_result is False, "FAKE_HASH must not exist on PureChain!"

    # Retrieve one real hash from tx_registry.json
    tx_reg_path = RESULTS_DIR / "../data/synthetic/tx_registry.json"
    if tx_reg_path.exists():
        with open(tx_reg_path) as f:
            tx_reg = json.load(f)
        real_hash = next(iter(tx_reg))
        real_result = client.verify_tx_hash(real_hash)
        print(f"  verify_tx_hash(real_hash)  → {real_result}  [{real_hash[:16]}…]")
        assert real_result is True, "Real hash must exist on PureChain!"
    else:
        print("  tx_registry.json not found — skipping real-hash check")
        real_hash = None

    print()
    print("  ✓ FAKE_HASH rejected (not on-chain)")
    if real_hash:
        print("  ✓ Real hash accepted (confirmed on-chain)")

    # ── Step 2: Audit tamper_results.json ──────────────────────
    tamper_path = RESULTS_DIR / "tamper_results.json"
    if not tamper_path.exists():
        print("\ntamper_results.json not found — run run_tamper_experiment.py first.")
        return

    with open(tamper_path) as f:
        records = json.load(f)

    scenarios_path = Path("data/synthetic/scenarios.json")
    with open(scenarios_path) as f:
        scenarios = json.load(f)
    scenario_map = {s["scenario_id"]: s for s in scenarios}

    print(f"\n=== Auditing {len(records)} tamper records ===\n")

    # For each C3 record, re-verify tamper_alerts via live blockchain calls
    # We reconstruct candidate record lists exactly as _build_tamper_contexts does:
    #   targeted → first asset (reorder) or first routing (others) has FAKE_HASH
    #   high     → all asset (reorder) or all routing (others) have FAKE_HASH

    updated = 0
    mismatches = 0
    detect_by_level: dict = defaultdict(lambda: {"total": 0, "detected": 0})

    for rec in records:
        if rec["condition"] != "C3" or rec["tamper_level"] == "baseline":
            continue

        sid    = rec["scenario_id"]
        level  = rec["tamper_level"]
        stype  = rec["scenario_type"]
        scenario = scenario_map.get(sid)
        if scenario is None:
            continue

        # Build the same candidate hash list as _build_tamper_contexts
        asset_hashes   = [r.get("chain_tx_hash", "") for r in scenario.get("asset_records",   [])]
        routing_hashes = [r.get("chain_tx_hash", "") for r in scenario.get("routing_records", [])]

        if level == "targeted":
            if stype == "reorder" and asset_hashes:
                cand_hashes = [FAKE_HASH] + asset_hashes[1:] + routing_hashes
                n_expected  = 1
            elif routing_hashes:
                cand_hashes = asset_hashes + [FAKE_HASH] + routing_hashes[1:]
                n_expected  = 1
            else:
                continue
        else:  # "high"
            if stype == "reorder":
                cand_hashes = [FAKE_HASH] * len(asset_hashes) + routing_hashes
                n_expected  = len(asset_hashes)
            else:
                cand_hashes = asset_hashes + [FAKE_HASH] * len(routing_hashes)
                n_expected  = len(routing_hashes)

        # Call verify_tx_hash for every candidate hash
        alerts = sum(1 for h in cand_hashes if not client.verify_tx_hash(h))

        old_alerts = rec["tamper_alerts"]
        rec["tamper_alerts"] = alerts
        updated += 1

        detect_by_level[level]["total"]    += n_expected
        detect_by_level[level]["detected"] += alerts

        if alerts != old_alerts:
            mismatches += 1
            print(f"  MISMATCH  {sid[:12]}  level={level}  "
                  f"type={stype}  old={old_alerts}  new={alerts}")
        elif alerts == 0 and n_expected > 0:
            print(f"  WARNING   {sid[:12]}  level={level}  "
                  f"type={stype}  expected {n_expected} alerts but got 0")

    # ── Step 3: Save updated records ───────────────────────────
    with open(tamper_path, "w") as f:
        json.dump(records, f, indent=2)

    # ── Step 4: Detection rate summary ─────────────────────────
    print(f"\n=== Detection Audit Summary ===\n")
    print(f"  Records updated:  {updated}")
    print(f"  Alert mismatches: {mismatches}  (0 expected)\n")

    for level, stats in sorted(detect_by_level.items()):
        rate = stats["detected"] / stats["total"] if stats["total"] else 0
        print(f"  {level:10s}  tampered={stats['total']:4d}  "
              f"detected={stats['detected']:4d}  rate={rate:.4f}")

    print()
    all_detected = all(
        s["detected"] == s["total"] for s in detect_by_level.values()
    )
    if all_detected:
        print("  ✓ C3 achieves 100% tamper detection (all FAKE_HASH records rejected)")
    else:
        print("  ✗ Detection is not 100% — investigate mismatches above")

    print(f"\n  Saved updated tamper_alerts → {tamper_path}")


if __name__ == "__main__":
    main()
