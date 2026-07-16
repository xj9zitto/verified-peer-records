"""
v14_content.py
Layer 2 — canonical serialization and content digests for v14 peer records.

Shared by the anchoring script (scripts/anchor_v14_peer_digests.py), the
content-bound verification predicate, and chain-based recovery, so that
all three hash byte-identical payloads.

The canonical payload deliberately excludes the provenance handle
(chain_tx_hash) and bookkeeping flags: the digest binds the record
CONTENT (identity, state snapshot, realized outcome), while the handle
is the pointer being verified.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any, Dict

PAYLOAD_FIELDS = ("peer_run_id", "scenario", "cause_family", "snapshot_at_s")


def peer_key(peer: Dict[str, Any]) -> str:
    """Stable identity key for a peer record (same scheme as
    assign_v14_chain_hashes.py)."""
    return (
        f"{peer.get('peer_run_id', '')}|"
        f"{peer.get('scenario', '')}|"
        f"{peer.get('snapshot_at_s', '')}"
    )


def canonical_peer_payload(peer: Dict[str, Any]) -> str:
    """Deterministic JSON serialization of a peer record's content."""
    snap = peer.get("snapshot", {}) or {}
    out = peer.get("outcome", {}) or {}
    payload = {
        "peer_run_id": str(peer.get("peer_run_id", "")),
        "scenario": peer.get("scenario", ""),
        "cause_family": peer.get("cause_family", -1),
        "snapshot_at_s": peer.get("snapshot_at_s", 0),
        "snapshot": {k: snap[k] for k in sorted(snap)},
        "outcome": {k: out[k] for k in sorted(out)},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def peer_content_digest(peer: Dict[str, Any]) -> str:
    """64-char lowercase hex SHA-256 digest of the canonical payload."""
    return hashlib.sha256(canonical_peer_payload(peer).encode()).hexdigest()
