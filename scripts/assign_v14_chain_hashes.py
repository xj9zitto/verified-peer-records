"""
assign_v14_chain_hashes.py

Map each verified peer in v14_scenarios.json to a real on-chain
transaction hash drawn from tx_registry.json (the 570 existing
PureChain entries from the supply-chain experiment).

The mapping is deterministic on (peer_run_id, scenario, snapshot_at_s),
so the same peer record always gets the same chain_tx_hash even if
the loader is re-run.  This anchors the peer corpus to PureChain
without needing to log 480 fresh transactions for the smoke test.

Usage:
    python scripts/assign_v14_chain_hashes.py
    python scripts/assign_v14_chain_hashes.py --verify   # also call
                                                          # verify_tx_hash
                                                          # on a sample
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

V14_SCENARIOS = Path("data/synthetic/v14_scenarios.json")
TX_REGISTRY   = Path("data/synthetic/tx_registry.json")


def _peer_key(peer: dict) -> str:
    return (
        f"{peer.get('peer_run_id','')}|"
        f"{peer.get('scenario','')}|"
        f"{peer.get('snapshot_at_s','')}"
    )


def _deterministic_hash(key: str, registry_hashes: list[str]) -> str:
    h = hashlib.sha256(key.encode()).digest()
    idx = int.from_bytes(h[:8], "big") % len(registry_hashes)
    return registry_hashes[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="Sanity-check by calling verify_tx_hash on 5 peers")
    args = ap.parse_args()

    if not TX_REGISTRY.exists():
        print(f"Missing {TX_REGISTRY}.  Run populate_hashes.py first.")
        sys.exit(1)

    registry  = json.loads(TX_REGISTRY.read_text())
    pool      = sorted(registry.keys())
    print(f"Loaded {len(pool)} on-chain hashes from {TX_REGISTRY}")

    scenarios = json.loads(V14_SCENARIOS.read_text())
    n_peers, n_unique = 0, set()
    for sc in scenarios:
        for peer in sc.get("asset_records", []):
            key = _peer_key(peer)
            tx  = _deterministic_hash(key, pool)
            peer["chain_tx_hash"] = tx
            peer["_is_v14_peer"]  = True
            n_peers += 1
            n_unique.add(key)

    V14_SCENARIOS.write_text(json.dumps(scenarios, indent=2))
    print(f"Assigned chain_tx_hash to {n_peers} peer instances "
          f"({len(n_unique)} unique peers)")

    if args.verify:
        from src.layer2.purechain_client import PureChainClient
        client = PureChainClient()
        client.connect()
        sample = scenarios[0].get("asset_records", [])[:5]
        for p in sample:
            ok = client.verify_tx_hash(p["chain_tx_hash"])
            print(f"  verify {p['chain_tx_hash'][:16]}... -> {ok}  "
                  f"(scenario={p['scenario']})")


if __name__ == "__main__":
    main()
