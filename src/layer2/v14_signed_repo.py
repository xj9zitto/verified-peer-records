"""
v14_signed_repo.py — an authenticated recovery store that is not a ledger.

Reviewer 2 observed that Table III conflated two things: the choice of
integrity mechanism, and whether an authenticated copy of the payload
was retained at all.  Committing payloads on-chain provides both, so
"only the ledger reconstructs" compared a ledger that keeps payloads
against signatures and Merkle proofs that keep none.  A signature-based
repository retaining authenticated payloads would also reconstruct.

This module is that baseline, built to be as strong as the on-chain
path and to differ from it in exactly one respect: who holds the
recovery copy.

  Detection   identical in strength.  A record is accepted only if its
              canonical payload carries a valid ECDSA signature from the
              anchoring identity, so tampering is caught whatever the
              handle says, exactly as content binding catches it.

  Recovery    the authenticated payload is served by one repository
              operator.  Under ``compromised=True`` that operator is
              inside the same breach as the local store and withholds
              the payload, which a verifier cannot distinguish from the
              record never having existed.  The ledger's validator set
              replicates the payload by consensus instead, which is
              what the comparison is meant to isolate.

The signature is held in the repository rather than beside the local
record, which is the baseline's strongest form: an adversary who edits
the store cannot strip a signature it never held.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

from eth_account import Account
from eth_account.messages import encode_defunct

from src.layer2.v14_content import canonical_peer_payload, peer_key

REPO_PATH = Path("data/synthetic/v14_signed_repository.json")


class SignedRepository:
    """Signature-authenticated payload store, ledger-free."""

    def __init__(
        self,
        path:           Path = REPO_PATH,
        signer_address: Optional[str] = None,
        compromised:    bool = False,
    ) -> None:
        blob = json.loads(Path(path).read_text())
        self.signer = (signer_address or blob["signer"]).lower()
        self.store: Dict[str, Dict[str, str]] = blob["records"]
        self.compromised = compromised
        self.stats = {"checked": 0, "rejected": 0,
                      "recovered": 0, "withheld": 0}

    # -- internals ------------------------------------------------
    def _sig_signer(self, payload: str, signature: str) -> Optional[str]:
        try:
            return Account.recover_message(
                encode_defunct(text=payload), signature=signature).lower()
        except Exception:
            return None

    def _authentic(self, entry: Dict[str, str]) -> bool:
        return self._sig_signer(entry["payload"],
                                entry["signature"]) == self.signer

    # -- predicates -----------------------------------------------
    def verify(self, peer: Dict[str, Any]) -> bool:
        """Accept the local record only if its payload is signed."""
        self.stats["checked"] += 1
        entry = self.store.get(peer_key(peer))
        if entry is None or entry["payload"] != canonical_peer_payload(peer):
            self.stats["rejected"] += 1
            return False
        if not self._authentic(entry):
            self.stats["rejected"] += 1
            return False
        return True

    def recover(self, peer: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Re-fetch the authenticated payload, unless the operator is
        inside the breach."""
        if self.compromised:
            self.stats["withheld"] += 1
            return None
        key = peer_key(peer)
        entry = self.store.get(key)
        if entry is None or not self._authentic(entry):
            return None
        try:
            rec = json.loads(entry["payload"])
        except Exception:
            return None
        if peer_key(rec) != key:
            return None
        rec["_is_v14_peer"] = True
        rec["_recovered_from_signed_repo"] = True
        self.stats["recovered"] += 1
        return rec
