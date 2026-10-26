"""
build_signed_repository.py

Builds the signature-authenticated payload repository used as the
ledger-free recovery baseline (Section IV-E, Table III).

Every unique peer record in the evaluation corpora is serialised with
the same canonical form the on-chain anchoring uses, then signed with a
dedicated anchoring key.  The repository therefore holds exactly the
information the ledger holds -- payload plus an authentication tag --
and differs only in who serves it.

Output: data/synthetic/v14_signed_repository.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eth_account import Account
from eth_account.messages import encode_defunct

from src.layer2.v14_content import canonical_peer_payload, peer_key

# A repository key distinct from the chain anchoring key: the baseline is
# a separate deployment, not a re-use of the ledger's identity.  Fixed so
# the artifact is reproducible.
REPO_KEY = "0x" + "a5" * 32
OUT = Path("data/synthetic/v14_signed_repository.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, nargs="+",
                    default=[Path("data/synthetic/v14_scenarios.json"),
                             Path("data/synthetic/v14_scenarios_expanded.json")])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    acct = Account.from_key(REPO_KEY)
    records = {}
    for path in args.scenarios:
        if not path.exists():
            print(f"  skip (absent): {path}")
            continue
        for sc in json.loads(path.read_text()):
            for p in sc.get("asset_records", []) or []:
                k = peer_key(p)
                if k in records:
                    continue
                payload = canonical_peer_payload(p)
                sig = Account.sign_message(
                    encode_defunct(text=payload), private_key=acct.key)
                records[k] = {"payload": payload,
                              "signature": sig.signature.hex()}
        print(f"  {path.name}: cumulative {len(records)} unique peers")

    args.out.write_text(json.dumps(
        {"signer": acct.address.lower(),
         "scheme": "ECDSA secp256k1 over EIP-191 personal_sign of the "
                   "canonical payload",
         "n_records": len(records),
         "records": records}, indent=2))
    print(f"\nSigner: {acct.address}")
    print(f"Saved -> {args.out}  ({len(records)} records)")


if __name__ == "__main__":
    main()
