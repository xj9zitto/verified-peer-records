"""
test_r2_verification.py — round-2 verifier hardening (IoT-74062-2026).

Covers, without any network access:
  * the multi-endpoint acceptance rule (unanimity by default, m-of-M on
    request) and the matching recovery read;
  * the mapping of RPC outcomes onto "failure" (evidence against a
    record) versus "unavailable" (no evidence at all);
  * the fail-closed path of decide_v14 and the recovery notice text.

Run:  python src/tests/test_r2_verification.py
"""

from __future__ import annotations
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from web3.exceptions import TransactionNotFound

from src.layer2.purechain_client import (
    PureChainClient, VerificationPolicy, VerificationUnavailable,
    MockPureChainClient, MIN_CONFIRMATIONS_DEFAULT,
)
from src.layer2.v14_content import peer_content_digest
import src.layer3.llm_agent_v14 as A
from src.layer1.v14_rule_engine import c0_decide_v14

H = "ab" * 32
D = "cd" * 32
D_BAD = "ef" * 32
CONTRACT = "0x" + "11" * 20
WRITER = "0x" + "22" * 20


# ── quorum ─────────────────────────────────────────────────────

class _Stub:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def get_peer_payload(self, h):
        if self.behaviour == "ok":
            return ("payload", D)
        if self.behaviour == "disagree":
            return ("payload", D_BAD)
        if self.behaviour == "none":
            return None
        raise VerificationUnavailable("down")


class _QClient(PureChainClient):
    """Primary answers 'ok'; extra endpoints follow ``plan``."""

    def __init__(self, plan, rule="unanimous", min_agree=2):
        super().__init__(rpc_url="primary", contract_address=CONTRACT,
                         policy=VerificationPolicy(quorum_rule=rule,
                                                   quorum_min_agree=min_agree),
                         quorum_urls=list(plan))
        self.plan = plan

    def get_peer_payload(self, h):
        return ("payload", D)

    def _peer_client(self, url):
        return _Stub(self.plan[url])


def test_quorum_single_endpoint_accepts():
    c = _QClient({})
    acc, agree, reach, conf, failed = c.verify_content_quorum(H, D)
    assert (acc, agree, reach, conf, failed) == (True, 1, 1, 1, [])


def test_quorum_unanimous_all_agree():
    c = _QClient({"e2": "ok", "e3": "ok"})
    assert c.verify_content_quorum(H, D)[0] is True
    assert c.verify_content(H, D) is True


def test_quorum_unanimous_rejects_unreachable():
    c = _QClient({"e2": "down"})
    acc, agree, reach, conf, failed = c.verify_content_quorum(H, D)
    assert acc is False and failed == ["e2"] and conf == 2 and reach == 1


def test_quorum_m_of_m_tolerates_unreachable():
    c = _QClient({"e2": "down"}, rule="m_of_m", min_agree=1)
    assert c.verify_content_quorum(H, D)[0] is True


def test_quorum_disagreement_rejects_under_both_rules():
    for rule in ("unanimous", "m_of_m"):
        c = _QClient({"e2": "disagree"}, rule=rule, min_agree=2)
        assert c.verify_content_quorum(H, D)[0] is False, rule


def test_quorum_recovery_read_requires_identical_answers():
    assert _QClient({"e2": "ok"}).get_peer_payload_quorum(H) == ("payload", D)
    assert _QClient({"e2": "disagree"}).get_peer_payload_quorum(H) is None
    assert _QClient({"e2": "down"}).get_peer_payload_quorum(H) is None
    assert _QClient({"e2": "none"}).get_peer_payload_quorum(H) is None


# ── endpoint-check error mapping ──────────────────────────────

class _FakeEth:
    def __init__(self, mode, fn="logAssetState", marker="V14PEER",
                 sender=WRITER, block=100, head=200):
        self.mode, self.fn, self.marker = mode, fn, marker
        self.sender, self.block, self._head = sender, block, head

    @property
    def block_number(self):
        return self._head

    def get_transaction_receipt(self, h):
        if self.mode == "missing":
            raise TransactionNotFound("no")
        if self.mode == "down":
            raise ConnectionError("endpoint down")
        return {"status": 1, "to": CONTRACT, "from": self.sender,
                "blockNumber": self.block}

    def get_transaction(self, h):
        return {"input": "0x"}


class _FakeContract:
    def __init__(self, fn, marker):
        self.fn, self.marker = fn, marker

    def decode_function_input(self, data):
        return (SimpleNamespace(fn_name=self.fn),
                {"hubLocation": self.marker, "assetId": "{}",
                 "sensorHash": bytes.fromhex(D)})


def _client(mode="ok", **kw):
    c = PureChainClient(rpc_url="x", contract_address=CONTRACT,
                        policy=VerificationPolicy(expected_writer=WRITER))
    eth = _FakeEth(mode, **kw)
    c.w3 = SimpleNamespace(eth=eth)
    c.contract = _FakeContract(eth.fn, eth.marker)
    c._connected = True
    return c


def test_defaults_enable_finality_and_function_check():
    p = VerificationPolicy()
    assert p.min_confirmations == MIN_CONFIRMATIONS_DEFAULT > 0
    assert p.check_function is True and p.quorum_rule == "unanimous"


def test_genuine_handle_passes():
    assert _client().endpoint_checks(H) == (True, [])


def test_missing_receipt_is_a_failure_not_an_outage():
    assert _client("missing").endpoint_checks(H) == (False, ["receipt_missing"])


def test_transport_error_raises_unavailable():
    try:
        _client("down").endpoint_checks(H)
    except VerificationUnavailable:
        return
    raise AssertionError("transport error was not surfaced as unavailable")


def test_malformed_handle_is_a_failure():
    assert _client().endpoint_checks("zz" * 32) == (False, ["malformed_handle"])
    assert _client().endpoint_checks("") == (False, ["empty_handle"])


def test_shallow_block_fails_finality():
    c = _client(block=198, head=200)
    assert c.endpoint_checks(H) == (False, ["insufficient_finality"])


def test_wrong_writer_rejected():
    c = _client(sender="0x" + "33" * 20)
    assert c.endpoint_checks(H) == (False, ["unauthorised_writer"])


def test_non_peer_call_rejected_in_handle_path():
    assert _client(fn="logDecision").endpoint_checks(H) == (
        False, ["not_peer_event"])
    assert _client(marker="HUB-1").endpoint_checks(H) == (
        False, ["not_peer_event"])


class _RecvEth(_FakeEth):
    """Receipt with an overridable recipient / status."""

    def __init__(self, to=CONTRACT, status=1, **kw):
        super().__init__("ok", **kw)
        self._to, self._status = to, status

    def get_transaction_receipt(self, h):
        return {"status": self._status, "to": self._to, "from": WRITER,
                "blockNumber": self.block}


def _client_eth(eth):
    c = PureChainClient(rpc_url="x", contract_address=CONTRACT,
                        policy=VerificationPolicy(expected_writer=WRITER))
    c.w3 = SimpleNamespace(eth=eth)
    c.contract = _FakeContract(eth.fn, eth.marker)
    c._connected = True
    return c


def test_wrong_contract_rejected():
    c = _client_eth(_RecvEth(to="0x" + "44" * 20))
    assert c.endpoint_checks(H) == (False, ["wrong_recipient"])


def test_reverted_transaction_rejected():
    c = _client_eth(_RecvEth(status=0))
    assert c.endpoint_checks(H) == (False, ["tx_reverted"])


def test_chain_id_mismatch_refuses_connection():
    class _W3:
        def __init__(self, *a, **k):
            self.eth = SimpleNamespace(chain_id=1)

        def is_connected(self):
            return True
    import src.layer2.purechain_client as pcm
    orig = pcm.Web3
    try:
        pcm.Web3 = type("W", (), {"HTTPProvider": staticmethod(lambda u: None),
                                  "__new__": lambda cls, prov: _W3()})
        c = PureChainClient(rpc_url="x", contract_address=CONTRACT,
                            policy=VerificationPolicy(expected_chain_id=900520900520))
        assert c.connect() is False
        assert c.verification_available() is False
    finally:
        pcm.Web3 = orig


def test_unconnected_client_is_unavailable():
    c = PureChainClient(rpc_url="x")
    assert c.verification_available() is False
    assert MockPureChainClient().verification_available() is False


# ── decide_v14 fail-closed and notice text ────────────────────

SCEN = json.loads(Path("data/synthetic/v14_scenarios.json")
                  .read_text(encoding="utf-8"))[0]
VALID = ("Reasoning: test.\nPredicted stage: 1\nStage-2 in 5 min: yes\n"
         "Action: replace\n")


class _PC:
    """Verifier stub: ``mode`` in ok|reject|down; ``recover`` in
    all|none|one."""

    def __init__(self, mode="ok", recover="all", available=True):
        self.mode, self.recover, self.available = mode, recover, available
        self.n_recovered = 0

    def verification_available(self):
        return self.available

    def verify_tx_hash(self, h):
        if self.mode == "down":
            raise VerificationUnavailable("down")
        return self.mode == "ok"

    def verify_content(self, h, d):
        return self.verify_tx_hash(h)

    def recover_payload(self, h):
        if self.mode == "down":
            raise VerificationUnavailable("down")
        if self.recover == "none":
            return None
        if self.recover == "one" and self.n_recovered >= 1:
            return None
        peer = next((p for p in SCEN["asset_records"]
                    if p.get("chain_tx_hash") == h), None)
        if peer is None:
            return None
        from src.layer2.v14_content import canonical_peer_payload as canonical_payload
        payload = canonical_payload(peer)
        import hashlib
        self.n_recovered += 1
        return payload, hashlib.sha256(payload.encode()).hexdigest()

    get_peer_payload = recover_payload


def _agent(pc):
    prompts = []

    def call(prompt):
        prompts.append(prompt)
        return VALID
    ag = SimpleNamespace(pc_client=pc, model="stub",
                         _is_anthropic_model=lambda: False,
                         _is_openai_model=lambda: False,
                         _call_ollama=call)
    return ag, prompts


def test_failclosed_when_unavailable_c3_c4():
    rule = c0_decide_v14(SCEN)["decision"]
    for cond in ("C3", "C4"):
        for pc in (_PC(mode="down"), _PC(available=False)):
            ag, prompts = _agent(pc)
            r = A.decide_v14(ag, copy.deepcopy(SCEN), condition=cond,
                             c3_recovery="drop")
            assert r["decision_source"] == "rule_engine_failclosed", cond
            assert r["decision"] == rule and prompts == [], cond


def test_c1_c2_do_not_need_verifier():
    for cond in ("C1", "C2"):
        ag, prompts = _agent(_PC(available=False))
        r = A.decide_v14(ag, copy.deepcopy(SCEN), condition=cond)
        assert r["decision_source"] == "llm" and len(prompts) == 1


def test_c3_genuine_rejects_still_use_llm():
    ag, prompts = _agent(_PC(mode="reject"))
    r = A.decide_v14(ag, copy.deepcopy(SCEN), condition="C3",
                     c3_recovery="drop")
    assert r["decision_source"] == "llm"
    assert r["tamper_alerts"] == len(SCEN["asset_records"])
    assert "were withheld" in prompts[0]


def test_c4_notice_matches_what_happened():
    full = ("and were excluded; canonical records were re-fetched "
            "from PureChain.")
    ag, prompts = _agent(_PC(mode="reject", recover="all"))
    r = A.decide_v14(ag, copy.deepcopy(SCEN), condition="C4")
    assert r["peers_recovered"] == r["tamper_alerts"] > 0
    assert full in prompts[0]

    ag, prompts = _agent(_PC(mode="reject", recover="none"))
    r = A.decide_v14(ag, copy.deepcopy(SCEN), condition="C4")
    assert r["peers_recovered"] == 0
    assert "re-fetched" not in prompts[0] and "were withheld" in prompts[0]

    ag, prompts = _agent(_PC(mode="reject", recover="one"))
    r = A.decide_v14(ag, copy.deepcopy(SCEN), condition="C4")
    assert r["peers_recovered"] == 1
    assert "the rest were withheld" in prompts[0]


def test_nominal_prompts_identical_across_c2_c3_c4():
    texts = []
    for cond in ("C2", "C3", "C4"):
        ag, prompts = _agent(_PC(mode="ok"))
        A.decide_v14(ag, copy.deepcopy(SCEN), condition=cond,
                     c3_recovery="drop")
        texts.append(prompts[0])
    assert texts[0] == texts[1] == texts[2]


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("  PASS  %s" % fn.__name__)
        except Exception:
            failed += 1
            print("  FAIL  %s" % fn.__name__)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
