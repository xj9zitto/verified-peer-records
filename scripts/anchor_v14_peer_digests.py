"""
anchor_v14_peer_digests.py

D1 (content-bound verification): anchor every unique v14 peer record on
PureChain with its canonical payload and SHA-256 content digest.

For each unique peer (identified by peer_run_id|scenario|snapshot_at_s):
  1. Serialize the canonical payload (src/layer2/v14_content.py).
  2. Log it on-chain via logAssetState with the payload in the assetId
     string field and the digest in the bytes32 sensorHash field.
  3. Record the new tx hash + digest in data/synthetic/v14_tx_registry.json.

Then every peer reference in v14_scenarios.json is updated in place:
  chain_tx_hash  -> the peer's own anchoring transaction
  content_digest -> the canonical content digest

A one-time backup of v14_scenarios.json is written before modification.

Usage:
    python scripts/anchor_v14_peer_digests.py            # anchor + update
    python scripts/anchor_v14_peer_digests.py --verify   # sample-check
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.layer2.v14_content import (
    canonical_peer_payload, peer_content_digest, peer_key,
)
from src.layer2.purechain_client import PureChainClient

V14_SCENARIOS = Path("data/synthetic/v14_scenarios.json")
BACKUP        = Path("data/synthetic/v14_scenarios.backup_pre_anchor.json")
REGISTRY      = Path("data/synthetic/v14_tx_registry.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="Sample-check verify_content on 5 anchored peers")
    ap.add_argument("--scenarios", type=Path, default=V14_SCENARIOS,
                    help="Scenario file whose peers are anchored.  The "
                         "registry is shared across files and keyed by "
                         "peer identity, so peers already anchored under "
                         "identical content are reused rather than "
                         "re-anchored.")
    args = ap.parse_args()

    scen_path = args.scenarios
    backup_path = scen_path.with_suffix(".backup_pre_anchor.json")
    scenarios = json.loads(scen_path.read_text(encoding="utf-8"))

    # collect unique peers; assert content is consistent across references
    unique: dict[str, dict] = {}
    n_refs = 0
    for sc in scenarios:
        for p in sc.get("asset_records", []):
            k = peer_key(p)
            n_refs += 1
            d = peer_content_digest(p)
            if k in unique:
                assert peer_content_digest(unique[k]) == d, \
                    f"Inconsistent content for peer {k}"
            else:
                unique[k] = p
    print(f"{n_refs} peer references, {len(unique)} unique peers to anchor")

    client = PureChainClient()
    if not client.connect():
        sys.exit("Cannot connect to PureChain")

    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    t0 = time.perf_counter()
    n_new = 0
    for i, (k, p) in enumerate(sorted(unique.items()), 1):
        digest = peer_content_digest(p)
        if k in registry and registry[k].get("digest") == digest:
            continue  # already anchored with identical content
        payload = canonical_peer_payload(p)
        snap = p.get("snapshot", {})
        tx = client.log_v14_peer(
            payload, digest,
            soc_bp=int(float(snap.get("SOC", 0)) * 10_000),
            temp_c=int(float(snap.get("temperature_C", 0)) * 100),
            stage=int(p.get("outcome", {}).get("failure_stage_max", 0)),
        )
        registry[k] = {"tx_hash": tx.replace("0x", ""), "digest": digest}
        n_new += 1
        if i % 25 == 0 or i == len(unique):
            REGISTRY.write_text(json.dumps(registry, indent=1))
            print(f"  [{i}/{len(unique)}] anchored "
                  f"({time.perf_counter()-t0:.1f}s elapsed)")
    REGISTRY.write_text(json.dumps(registry, indent=1))
    print(f"Anchored {n_new} new peers in {time.perf_counter()-t0:.1f}s "
          f"-> {REGISTRY}")

    # update scenario peer references
    if not backup_path.exists():
        backup_path.write_text(scen_path.read_text(encoding="utf-8"),
                               encoding="utf-8")
        print(f"Backup written -> {backup_path}")
    n_upd = 0
    for sc in scenarios:
        for p in sc.get("asset_records", []):
            entry = registry[peer_key(p)]
            p["chain_tx_hash"]  = entry["tx_hash"]
            p["content_digest"] = entry["digest"]
            n_upd += 1
    scen_path.write_text(json.dumps(scenarios, indent=2),
                         encoding="utf-8")
    print(f"Updated {n_upd} peer references in {scen_path}")

    if args.verify:
        sample = scenarios[0]["asset_records"][:3] + \
                 scenarios[-1]["asset_records"][:2]
        for p in sample:
            ok = client.verify_content(p["chain_tx_hash"],
                                       peer_content_digest(p))
            got = client.get_peer_payload(p["chain_tx_hash"])
            rt = (got is not None
                  and json.loads(got[0]).get("peer_run_id")
                      == str(p["peer_run_id"]))
            print(f"  {p['chain_tx_hash'][:16]}...  "
                  f"verify_content={ok}  payload_roundtrip={rt}")


if __name__ == "__main__":
    main()
