"""
validate_endpoint_checks.py

Exercises the thin-client verifier against live PureChain state, under
the round-2 policy (pinned writer, finality depth, function check,
unanimity across endpoints).

Reports, for every anchored v14 peer handle in the registry:
  * genuine handles rejected under the default policy (should be 0),
    and the latency of each layer of checks;
  * negative controls, each of which must be rejected, and for the
    stated reason: a fabricated handle, a malformed handle, a successful
    call to the same contract that is not a peer anchor, and the
    round-1 anchors on the superseded contract;
  * a positive control for the writer check: under a policy naming a
    different writer, every genuine handle is rejected for that reason;
  * content-bound verification of every handle against its registry
    digest, and its latency;
  * the multi-endpoint rule: an alias of the same endpoint (agreement)
    and an alias plus an unreachable endpoint (must fail closed);
  * confirmation depth, and what the endpoint says about its sealers.

Output: results/endpoint_check_validation.json
"""

from __future__ import annotations
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.layer2.purechain_client import (
    PureChainClient, VerificationPolicy, VerificationUnavailable,
    MIN_CONFIRMATIONS_DEFAULT,
)

REG_FILE = Path("data/synthetic/v14_tx_registry.json")
OLD_REG = Path("data/synthetic/v14_tx_registry.contract_0x9fE4.json")
CONTROL_FILE = Path("data/synthetic/endpoint_controls.json")
OUT = Path("results/endpoint_check_validation.json")

FAKE_HASH = "0x" + "de" * 32
MALFORMED = "0x" + "zz" * 32
DEAD_ENDPOINT = "http://127.0.0.1:9"


def _p95(xs):
    return round(sorted(xs)[max(0, int(0.95 * len(xs)) - 1)], 2)


def _time_checks(pc, handles):
    lat, rejected = [], {}
    for h in handles:
        t0 = time.perf_counter()
        ok, why = pc.endpoint_checks(h)
        lat.append((time.perf_counter() - t0) * 1000)
        if not ok:
            rejected[h] = why
    return {"rejected": len(rejected),
            "rejected_detail": dict(list(rejected.items())[:5]),
            "latency_ms_mean": round(st.mean(lat), 2),
            "latency_ms_p95": _p95(lat)}


def _client(**policy):
    pc = PureChainClient(policy=VerificationPolicy(**policy))
    assert pc.connect(), "connect failed"
    return pc


def _non_peer_control(pc) -> str:
    """A successful logDecision call to the same contract by the same
    writer: it passes status, recipient and writer, so only the function
    check can reject it.  Created once and reused."""
    if CONTROL_FILE.exists():
        ctl = json.loads(CONTROL_FILE.read_text())
        if ctl.get("contract") == pc.contract_address:
            return ctl["log_decision_tx"]
    tx = pc.log_decision("control", "endpoint_check_control", 0.0)
    CONTROL_FILE.write_text(json.dumps(
        {"contract": pc.contract_address, "log_decision_tx": tx}, indent=1))
    return tx


def main() -> None:
    reg = json.loads(REG_FILE.read_text())
    handles = [v["tx_hash"] for v in reg.values() if v.get("tx_hash")]
    print("anchored peer handles: %d" % len(handles))

    pc = _client()
    report = {
        "n_handles": len(handles),
        "contract": pc.contract_address,
        "expected_writer": pc.policy.expected_writer,
        "writer_source": pc.writer_source,
        "policy": {"min_confirmations": pc.policy.min_confirmations,
                   "check_function": pc.policy.check_function,
                   "quorum_rule": pc.policy.quorum_rule},
    }

    # -- depth and sealers -------------------------------------
    head = pc.w3.eth.block_number
    depths = []
    for h in handles:
        r = pc.w3.eth.get_transaction_receipt(h)
        depths.append(head - int(r["blockNumber"]) + 1)
    report["confirmation_depth"] = {"min": min(depths),
                                    "median": int(st.median(depths)),
                                    "max": max(depths)}
    try:
        signers = pc.w3.provider.make_request("clique_getSigners", ["latest"])
        report["clique_getSigners"] = signers.get("result", signers.get("error"))
    except Exception as exc:
        report["clique_getSigners"] = "not exposed: %s" % exc
    print("  depth min %d  sealers %s" % (min(depths),
                                           report["clique_getSigners"]))

    # -- genuine handles, layer by layer -----------------------
    layers = {
        "receipt_status_recipient_writer":
            dict(min_confirmations=0, check_function=False),
        "plus_finality":
            dict(min_confirmations=MIN_CONFIRMATIONS_DEFAULT,
                 check_function=False),
        "default_full":
            dict(),
    }
    report["genuine"] = {}
    for name, pol in layers.items():
        res = _time_checks(_client(**pol), handles)
        report["genuine"][name] = res
        print("  %-34s rejected %d/%d  mean %.1f ms  p95 %.1f ms"
              % (name, res["rejected"], len(handles),
                 res["latency_ms_mean"], res["latency_ms_p95"]))

    # -- content-bound verification of every handle -------------
    lat, n_ok = [], 0
    for v in reg.values():
        t0 = time.perf_counter()
        n_ok += pc.verify_content(v["tx_hash"], v["digest"])
        lat.append((time.perf_counter() - t0) * 1000)
    report["content_verify"] = {"accepted": n_ok, "of": len(reg),
                                "latency_ms_mean": round(st.mean(lat), 2),
                                "latency_ms_p95": _p95(lat)}
    print("  content verify accepted %d/%d  mean %.1f ms"
          % (n_ok, len(reg), st.mean(lat)))

    # -- negative controls --------------------------------------
    controls = {"fabricated_handle": FAKE_HASH,
                "malformed_handle": MALFORMED,
                "non_peer_call_same_contract": _non_peer_control(pc)}
    report["negative_controls"] = {}
    for name, h in controls.items():
        ok, why = pc.endpoint_checks(h)
        report["negative_controls"][name] = {"accepted": ok, "failed": why}
        print("  %-30s accepted=%s %s" % (name, ok, why))

    if OLD_REG.exists():
        old = [v["tx_hash"] for v in json.loads(OLD_REG.read_text()).values()]
        reasons, n_rej = set(), 0
        for h in old:
            ok, why = pc.endpoint_checks(h)
            n_rej += not ok
            reasons.update(why)
        report["negative_controls"]["round1_anchors_old_contract"] = {
            "rejected": n_rej, "of": len(old), "reasons": sorted(reasons)}
        print("  round-1 anchors on the old contract rejected %d/%d %s"
              % (n_rej, len(old), sorted(reasons)))

    # -- writer positive control --------------------------------
    pc_wrong = _client(
        expected_writer="0x000000000000000000000000000000000000dEaD")
    reasons, n_rej = set(), 0
    for h in handles:
        ok, why = pc_wrong.endpoint_checks(h)
        n_rej += not ok
        reasons.update(why)
    report["writer_control"] = {"rejected_under_wrong_writer": n_rej,
                                "of": len(handles),
                                "reasons": sorted(reasons)}
    print("  wrong pinned writer: %d/%d rejected %s"
          % (n_rej, len(handles), sorted(reasons)))

    # -- multi-endpoint rule ------------------------------------
    alias = pc.rpc_url.rstrip("/") + "/"
    sample = list(reg.values())
    quorum = {}
    for name, extra in {"alias_agrees": [alias],
                        "alias_plus_unreachable": [alias, DEAD_ENDPOINT]
                        }.items():
        q = PureChainClient(quorum_urls=extra)
        assert q.connect()
        acc, unavailable = 0, 0
        for v in sample:
            try:
                acc += q.verify_content(v["tx_hash"], v["digest"])
            except VerificationUnavailable:
                unavailable += 1
        quorum[name] = {"endpoints": [q.rpc_url] + extra,
                        "accepted": acc, "of": len(sample),
                        "unavailable": unavailable}
        print("  quorum %-24s accepted %d/%d" % (name, acc, len(sample)))
    report["quorum_unanimous"] = quorum

    report["check_stats"] = dict(pc.check_stats)
    OUT.write_text(json.dumps(report, indent=2))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
