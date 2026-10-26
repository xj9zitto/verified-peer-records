"""
compare_integrity_mechanisms.py

Quantitative comparison of the ledger-anchored integrity layer against
the simpler mechanisms a reviewer would reach for first: plain digital
signatures over each record, and a tamper-evident (Merkle) append-only
log.

The comparison is on the four axes the review asked for -- trust
assumptions, storage, operational cost, and latency -- with every
latency and size figure measured on this machine and this deployment
rather than quoted.

What separates the three mechanisms is not cryptographic strength; all
three detect a modified record.  It is *who must be trusted for
availability and non-equivocation*:

  signatures        detect modification, but a compromised record store
                    can withhold a record or serve an older signed
                    version, and nothing in the signature reveals it.
                    Trust: the store operator.
  tamper-evident    append-only proofs make deletion and reordering
  log               detectable, provided verifiers can compare their
                    view of the log root.  Trust: a witness or auditor
                    set that publishes and cross-checks roots.
  ledger anchor     the witness set is the validator set, and the root
                    is replicated by consensus.  Trust: the validator
                    honest-majority assumption plus at least one honest
                    RPC endpoint.

Output: results/integrity_mechanism_comparison.json
"""

from __future__ import annotations
import hashlib
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eth_account import Account
from eth_account.messages import encode_defunct

from src.layer2.purechain_client import PureChainClient
from src.layer2.v14_content import canonical_peer_payload

SCEN_FILE = Path("data/synthetic/v14_scenarios.json")
REG_FILE  = Path("data/synthetic/v14_tx_registry.json")
OUT       = Path("results/integrity_mechanism_comparison.json")

N_TRIALS = 500


def _bench(fn, n=N_TRIALS):
    """Return (mean_ms, p95_ms) for n calls of fn."""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return round(st.mean(ts), 4), round(ts[int(0.95 * n) - 1], 4)


def merkle_root(leaves):
    level = [bytes.fromhex(h) for h in leaves]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(hashlib.sha256(a + b).digest())
        level = nxt
    return level[0]


def merkle_proof(leaves, index):
    level = [bytes.fromhex(h) for h in leaves]
    proof = []
    idx = index
    while len(level) > 1:
        sib = idx ^ 1
        if sib >= len(level):
            sib = idx
        proof.append((level[sib], idx % 2 == 1))
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(hashlib.sha256(a + b).digest())
        level = nxt
        idx //= 2
    return proof, level[0]


def verify_merkle(leaf_hex, proof, root):
    cur = bytes.fromhex(leaf_hex)
    for sib, right in proof:
        cur = (hashlib.sha256(sib + cur).digest() if right
               else hashlib.sha256(cur + sib).digest())
    return cur == root


def main() -> None:
    scen = json.loads(SCEN_FILE.read_text())
    peers = {}
    for s in scen:
        for p in s.get("asset_records", []):
            peers[(p.get("peer_run_id"), p.get("snapshot_at_s"))] = p
    peers = list(peers.values())
    payloads = [canonical_peer_payload(p) for p in peers]
    digests = [hashlib.sha256(x.encode()).hexdigest() for x in payloads]

    sizes = [len(x.encode()) for x in payloads]
    report = {
        "n_unique_peers": len(peers),
        "payload_bytes": {
            "mean": round(st.mean(sizes), 1),
            "min": min(sizes), "max": max(sizes),
            "total": sum(sizes),
        },
    }
    print("unique peers: %d   canonical payload: mean %.0f B (total %.1f KiB)"
          % (len(peers), st.mean(sizes), sum(sizes) / 1024))

    # -- 1. digest computation --------------------------------
    p0 = peers[0]
    m, p95 = _bench(lambda: hashlib.sha256(
        canonical_peer_payload(p0).encode()).hexdigest())
    report["digest_ms"] = {"mean": m, "p95": p95}
    print("  digest (serialize+SHA-256):   mean %.4f ms" % m)

    # -- 2. digital signature ---------------------------------
    acct = Account.create()
    msg = encode_defunct(text=payloads[0])
    sig = Account.sign_message(msg, private_key=acct.key)

    m, p95 = _bench(lambda: Account.recover_message(msg, signature=sig.signature))
    report["signature_verify_ms"] = {"mean": m, "p95": p95}
    report["signature_bytes_per_record"] = 65
    print("  ECDSA signature verify:       mean %.4f ms  (+65 B/record)" % m)

    # -- 2b. signature-authenticated payload repository -------
    # A reviewer noted that the original comparison credited the ledger
    # with reconstruction when the property really follows from retaining
    # an authenticated payload, which a signed repository also does.  The
    # arm is measured here so the comparison separates the integrity
    # mechanism from the recovery copy.  A verify is a signature check
    # plus the dictionary lookup that serves the payload; the recovery
    # path additionally parses it and re-checks identity.
    try:
        from src.layer2.v14_signed_repo import SignedRepository
        repo = SignedRepository()
        p_rec = peers[0]
        m, p95 = _bench(lambda: repo.verify(p_rec))
        report["signed_repo_verify_ms"] = {"mean": m, "p95": p95}
        m2, p952 = _bench(lambda: repo.recover(p_rec))
        report["signed_repo_recover_ms"] = {"mean": m2, "p95": p952}
        # 65 B signature plus the payload the repository must retain.
        report["signed_repo_bytes_per_record"] = 65 + round(st.mean(sizes))
        report["signed_repo_n_records"] = len(repo.store)
        print("  signed-repo verify:           mean %.4f ms  (+%d B/record)"
              % (m, report["signed_repo_bytes_per_record"]))
        print("  signed-repo recover:          mean %.4f ms" % m2)
    except Exception as exc:
        print("  signed repository: unavailable (%s)" % exc)

    # -- 3. tamper-evident log --------------------------------
    root = merkle_root(digests)
    proof, root2 = merkle_proof(digests, 0)
    assert root == root2
    m, p95 = _bench(lambda: verify_merkle(digests[0], proof, root))
    depth = len(proof)
    report["merkle"] = {
        "verify_ms": {"mean": m, "p95": p95},
        "proof_depth": depth,
        "proof_bytes_per_record": depth * 32,
        "root_bytes": 32,
    }
    print("  Merkle inclusion verify:      mean %.4f ms  "
          "(depth %d, +%d B/record)" % (m, depth, depth * 32))

    # -- 4. ledger anchor -------------------------------------
    reg = json.loads(REG_FILE.read_text())
    handles = [v["tx_hash"] for v in reg.values() if v.get("tx_hash")]
    pc = PureChainClient()
    pc.connect()

    h0 = handles[0]
    m, p95 = _bench(lambda: pc.verify_content(h0, digests[0]) or True, n=30)
    report["ledger"] = {
        "verify_ms": {"mean": m, "p95": p95},
        "n_anchored": len(handles),
        "handle_bytes_per_record": 32,
        "gas_price_wei": 0,
        "onchain_payload": "canonical payload + 32 B digest per record",
    }
    print("  ledger content verify (RPC):  mean %.1f ms  p95 %.1f ms"
          % (m, p95))

    # -- block interval, for finality cost --------------------
    try:
        head = pc.w3.eth.block_number
        # Raw RPC: PoA blocks carry a 97-byte extraData that the
        # standard block formatter rejects.
        def ts(n):
            r = pc.w3.provider.make_request(
                "eth_getBlockByNumber", [hex(n), False])
            return int(r["result"]["timestamp"], 16)
        # Block production on this PoA network is on demand rather than
        # periodic, so a single window is misleading: the interval
        # depends on how busy the chain is.  Report several windows.
        windows = {}
        for w in (1000, 10000, 50000, 93562):
            try:
                windows[str(w)] = round((ts(head) - ts(head - w)) / w, 1)
            except Exception:
                pass
        report["ledger"]["block_interval_s_by_window"] = windows
        report["ledger"]["block_interval_note"] = (
            "blocks are produced on demand; the interval is "
            "workload-dependent, from ~41 s/block averaged over the "
            "44.5 days spanning our anchoring campaign to ~1200 s/block "
            "while the chain is idle")
        print("  block interval by window (s): " + ", ".join(
            "%s:%s" % (k, v) for k, v in windows.items()))
    except Exception as exc:
        print("  block interval: unavailable (%s)" % exc)

    OUT.write_text(json.dumps(report, indent=2))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
