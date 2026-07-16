"""
scripts/populate_hashes.py
Log every record in scenarios.json to the real PureChain contract
(SupplyEventLogger.sol on Hardhat) and write the returned
chain_tx_hash back into the dataset.

Prerequisites:
  - Hardhat node running:   npx hardhat node
  - Contract deployed:      SUPPLY_EVENT_LOGGER_ADDRESS in .env
  - PURECHAIN_RPC_URL set in .env (already configured)

Run once after scenario generation:
    python scripts/populate_hashes.py

If a previous run left corrupted values:
    python scripts/populate_hashes.py --reset

WHY HARDHAT AND NOT THE REAL PURECHAIN PoA² NODE?
---------------------------------------------------
The PoA² node (port 8546) is used for TPS benchmarking and
production deployment. For the evaluation phase (120 scenarios ×
4 conditions), Hardhat provides identical contract semantics with
instant block mining, zero gas costs, and full web3.py compatibility.
The paper's contribution is the integration architecture; blockchain
performance figures come from published PureChain benchmarks.
"""

from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.layer2.purechain_client import get_client
from src.logger_setup import setup_logger

setup_logger("populate_hashes")

SCENARIOS_PATH = Path("data/synthetic/scenarios.json")
TX_REGISTRY    = Path("data/synthetic/tx_registry.json")


def _extract_tx(return_value) -> str:
    """
    Normalise the return value of log_* calls to a plain tx hash string.

    PureChainClient.log_asset_state() returns (tx_hash, event_id) — a
    tuple of two hex strings. We want only the tx_hash (first element).
    MockPureChainClient may also return a list or plain string.
    """
    if isinstance(return_value, (list, tuple)):
        value = return_value[0]
    else:
        value = return_value
    # Strip any accidental brackets/quotes from str(list) conversion
    s = str(value).strip().strip("[]'\"")
    return s


def reset_hashes(scenarios: list) -> list:
    """Clear all chain_tx_hash fields to empty string."""
    for s in scenarios:
        for rec in s.get("asset_records", []) + s.get("routing_records", []):
            rec["chain_tx_hash"] = ""
    return scenarios


def _is_valid_hash(value) -> bool:
    """Return True only if value is a 64-char lowercase hex string."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    return len(v) == 64 and all(c in "0123456789abcdef" for c in v.lower())


def populate(dry_run: bool = False, reset: bool = False) -> None:
    # Use real PureChain client — PURECHAIN_RPC_URL is set in .env
    client = get_client()

    if not client.is_connected:
        logger.error(
            "Cannot connect to PureChain node. "
            "Start Hardhat: npx hardhat node"
        )
        sys.exit(1)

    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)

    if reset:
        scenarios = reset_hashes(scenarios)
        logger.info("All chain_tx_hash fields reset to ''")

    tx_registry: dict[str, dict] = {}
    n_asset = n_routing = skipped = 0

    for s in scenarios:

        # ── Type-A: asset state records ───────────────────────
        for rec in s.get("asset_records", []):
            if _is_valid_hash(rec.get("chain_tx_hash", "")):
                skipped += 1
                continue  # already has a good hash

            result = _log_asset(client, rec)
            tx = _extract_tx(result)
            rec["chain_tx_hash"] = tx
            tx_registry[tx] = {
                "type":        "asset_state",
                "asset_id":    rec.get("asset_id", ""),
                "timestamp":   rec.get("timestamp", ""),
                "sensor_hash": rec.get("sensor_hash", ""),
            }
            n_asset += 1

        # ── Type-B: routing condition records ─────────────────
        for rec in s.get("routing_records", []):
            if _is_valid_hash(rec.get("chain_tx_hash", "")):
                skipped += 1
                continue

            result = _log_routing(client, rec)
            tx = _extract_tx(result)
            rec["chain_tx_hash"] = tx
            tx_registry[tx] = {
                "type":       "routing_condition",
                "path_id":    rec.get("path_id", ""),
                "hop_origin": rec.get("hop_origin", ""),
                "hop_dest":   rec.get("hop_dest", ""),
            }
            n_routing += 1

    logger.info(
        f"Done — asset={n_asset}  routing={n_routing}  "
        f"skipped={skipped}  total_new_txs={len(tx_registry)}"
    )

    if dry_run:
        logger.info("Dry run — nothing written to disk")
        return

    with open(SCENARIOS_PATH, "w") as f:
        json.dump(scenarios, f, indent=2)
    logger.success(f"scenarios.json updated — {SCENARIOS_PATH}")

    TX_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(TX_REGISTRY, "w") as f:
        json.dump(tx_registry, f, indent=2)
    logger.success(f"TX registry saved — {TX_REGISTRY} ({len(tx_registry)} hashes)")

    # Spot-check
    s0  = scenarios[0]
    a_tx = s0["asset_records"][0]["chain_tx_hash"]
    r_tx = s0["routing_records"][0]["chain_tx_hash"]
    logger.info(f"Sample asset tx    ({len(a_tx)} chars): {a_tx[:48]}...")
    logger.info(f"Sample routing tx  ({len(r_tx)} chars): {r_tx[:48]}...")
    logger.info(f"Valid 64-char hex: asset={_is_valid_hash(a_tx)}  routing={_is_valid_hash(r_tx)}")

    # Verify first hash actually exists on-chain
    if _is_valid_hash(a_tx):
        ok = client.verify_tx_hash(a_tx)
        logger.info(f"On-chain verification of sample: {ok}")


def _log_asset(client, rec: dict):
    from src.layer1.event_generator import AssetStateRecord
    return client.log_asset_state(AssetStateRecord(
        asset_id        = rec["asset_id"],
        timestamp       = rec["timestamp"],
        hub_location    = rec["hub_location"],
        soc_pct         = rec["soc_pct"],
        temperature_c   = rec["temperature_c"],
        degradation_pct = rec["degradation_pct"],
        sensor_hash     = rec["sensor_hash"],
        chain_tx_hash   = "",
    ))


def _log_routing(client, rec: dict):
    from src.layer1.event_generator import RoutingConditionRecord
    return client.log_routing_condition(RoutingConditionRecord(
        path_id            = rec["path_id"],
        hop_origin         = rec["hop_origin"],
        hop_dest           = rec["hop_dest"],
        transit_h          = rec["transit_h"],
        amb_temp_min_c     = rec["amb_temp_min_c"],
        amb_temp_max_c     = rec["amb_temp_max_c"],
        temp_controlled    = rec["temp_controlled"],
        carrier_id         = rec["carrier_id"],
        carrier_variance_c = rec["carrier_variance_c"],
        chain_tx_hash      = "",
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Log scenario records to PureChain and populate chain_tx_hash"
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Preview without writing to disk")
    p.add_argument("--reset", action="store_true",
                   help="Clear all existing hashes and re-log everything")
    args = p.parse_args()
    populate(dry_run=args.dry_run, reset=args.reset)