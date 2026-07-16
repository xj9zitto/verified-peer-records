"""
scripts/build_index.py
Build the persistent ChromaDB vector index from scenarios.json.

Prerequisites:
  - populate_hashes.py already run (chain_tx_hash populated)
  - Hardhat node running (for C3 verification during retrieval check)

Run once before experiments:
    python scripts/build_index.py

Rebuild from scratch:
    python scripts/build_index.py --rebuild
"""

from __future__ import annotations
import json
import sys
import argparse
import shutil
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.layer2.vector_index import VectorIndex
from src.layer2.purechain_client import get_client
from src.config import CHROMA_DIR
from src.logger_setup import setup_logger

setup_logger("build_index")

SCENARIOS_PATH = Path("data/synthetic/scenarios.json")
TX_REGISTRY    = Path("data/synthetic/tx_registry.json")


def check_hashes() -> tuple[int, int]:
    """Return (total_records, empty_hash_count)."""
    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)
    total = empty = 0
    for s in scenarios:
        for rec in s.get("asset_records", []) + s.get("routing_records", []):
            total += 1
            if not rec.get("chain_tx_hash", ""):
                empty += 1
    return total, empty


def build(rebuild: bool = False) -> None:
    chroma_dir = Path(CHROMA_DIR)

    if not SCENARIOS_PATH.exists():
        logger.error(f"scenarios.json not found at {SCENARIOS_PATH}")
        logger.error("Run: python src/dataset/scenario_gen.py")
        sys.exit(1)

    # Warn if hashes are missing
    total, empty = check_hashes()
    if empty > 0:
        logger.warning(
            f"{empty}/{total} records have empty chain_tx_hash. "
            "Run: python scripts/populate_hashes.py --reset"
        )
    else:
        logger.info(f"All {total} records have chain_tx_hash populated ✓")

    if rebuild and chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        logger.warning(f"Wiped existing index at {chroma_dir}")

    # Real PureChain client — uses PURECHAIN_RPC_URL from .env
    client = get_client()

    if not client.is_connected:
        logger.error(
            "PureChain node not reachable. "
            "Start Hardhat: npx hardhat node"
        )
        sys.exit(1)

    index = VectorIndex(client=client, persist_dir=str(chroma_dir))

    counts = index.collection_counts()
    if counts["asset_records"] > 0 and not rebuild:
        logger.info(
            f"Index already exists — "
            f"asset={counts['asset_records']} "
            f"routing={counts['routing_records']}. "
            "Use --rebuild to recreate."
        )
        _verify_retrieval(index)
        return

    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)

    logger.info(f"Indexing {len(scenarios)} scenarios...")
    n_asset = n_routing = 0
    for i, s in enumerate(scenarios):
        na, nr = index.add_scenario_records(s)
        n_asset   += na
        n_routing += nr
        if (i + 1) % 30 == 0:
            logger.info(f"  {i+1}/{len(scenarios)} scenarios indexed")

    counts = index.collection_counts()
    logger.success(
        f"Index built — "
        f"asset_records={counts['asset_records']}  "
        f"routing_records={counts['routing_records']}"
    )

    _verify_retrieval(index)


def _verify_retrieval(index: VectorIndex) -> None:
    """Quick retrieval smoke-test under C3 (full verification)."""
    result = index.retrieve(
        query="Should we reorder asset B-100 at H3?",
        k_asset=3,
        k_routing=2,
        verify=True,
    )
    logger.info(
        f"Retrieval check — "
        f"asset={len(result['asset_records'])}  "
        f"routing={len(result['routing_records'])}  "
        f"tamper_alerts={result['tamper_alerts']}"
    )
    if result["tamper_alerts"] == 0:
        logger.success("Retrieval check passed — 0 tamper alerts ✓")
    else:
        logger.warning(
            f"{result['tamper_alerts']} tamper alerts on retrieval check. "
            "Re-run populate_hashes.py --reset then rebuild."
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true",
                   help="Wipe and rebuild the index from scratch")
    args = p.parse_args()
    build(rebuild=args.rebuild)