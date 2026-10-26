"""
v14_transfer.py — the four verification arms of the transfer study,
shared by every experiment that swaps the decision architecture.

``run_verification_transfer.py`` established that the content-binding
effect survives replacing the LLM with a deterministic rule.  The
boundary-attack study asks a different question of the same arms: what
happens when the attack moves outside the content-integrity guarantee.
The two must run identical decision code for their results to be
comparable, so the arms and the chain cache live here rather than being
duplicated per runner.

The arms mirror the LLM conditions exactly:

  T0  observables only, no peer evidence              (analogue of C0)
  T2  peers consumed as retrieved, unverified         (analogue of C2)
  T3  handle-existence check, rejected peers dropped  (analogue of C3d)
  T4  content-bound verification + chain recovery     (analogue of C4)

Two further arms answer a reviewer who pointed out that Table III
conflated the integrity mechanism with the retention of a recovery copy.
Both use a signature-authenticated payload repository, which detects
tampering exactly as well as content binding and reconstructs from its
own authenticated copy, so they isolate who holds that copy:

  T5  signed-payload repository, operator honest
  T5x signed-payload repository, operator inside the same breach as the
      local store, so the authenticated payload is withheld
"""

from __future__ import annotations
from typing import Any, Dict, List

from src.layer1.v14_peer_vote import evidence_vote
from src.layer3.llm_agent_v14 import (
    _verify_peer_records, _verify_peer_records_content,
    _recover_peer_from_chain,
)

ARMS = ["T0", "T2", "T3", "T4"]
# Arms needing the ledger-free repository; opt-in, since they require the
# repository artifact to be built.
REPO_ARMS = ["T5", "T5x"]


class CachedChain:
    """
    Memoising proxy over the chain client.

    Committed transactions are immutable, so the answer to a given
    (handle) or (handle, digest) query cannot change during a run.
    Caching makes repeated seeds cheap without altering the
    verification semantics: every distinct query is still resolved
    against the live chain exactly once.
    """

    def __init__(self, pc):
        self.pc = pc
        self._tx, self._content, self._payload = {}, {}, {}
        self.calls = 0

    def verify_tx_hash(self, h):
        if h not in self._tx:
            self.calls += 1
            self._tx[h] = self.pc.verify_tx_hash(h)
        return self._tx[h]

    def verify_content(self, h, d):
        k = (h, d)
        if k not in self._content:
            self.calls += 1
            self._content[k] = self.pc.verify_content(h, d)
        return self._content[k]

    def get_peer_payload(self, h):
        if h not in self._payload:
            self.calls += 1
            self._payload[h] = self.pc.get_peer_payload(h)
        return self._payload[h]

    def recover_payload(self, h):
        reader = (getattr(self.pc, "recover_payload", None)
                  or self.pc.get_peer_payload)
        k = ("recover", h)
        if k not in self._payload:
            self.calls += 1
            self._payload[k] = reader(h)
        return self._payload[k]

    def verification_available(self):
        avail = getattr(self.pc, "verification_available", None)
        return bool(avail()) if avail else False


def decide(arm: str, scenario: Dict[str, Any], pc,
           repo=None, repo_compromised=None) -> Dict[str, Any]:
    """Run one arm on one (possibly poisoned) scenario.

    ``repo`` is a SignedRepository, required only by the T5 arms.  The
    same instance serves both, with the withholding behaviour selected
    per call so one artifact covers the honest and compromised operator.
    """
    peers = scenario.get("asset_records", []) or []
    tamper_alerts = 0
    recovered = 0

    if arm in ("T5", "T5x"):
        if repo is None:
            raise ValueError(f"{arm} requires a SignedRepository")
        repo.compromised = (arm == "T5x" if repo_compromised is None
                            else repo_compromised)
        use, rejected = [], []
        for p in peers:
            (use if repo.verify(p) else rejected).append(p)
        tamper_alerts = len(rejected)
        for p in rejected:
            r = repo.recover(p)
            if r is not None:
                use = use + [r]
                recovered += 1
    elif arm == "T0":
        use: List[Dict[str, Any]] = []
    elif arm == "T2":
        use = peers
    elif arm == "T3":
        use, n_rej = _verify_peer_records(peers, pc)
        tamper_alerts = n_rej
    elif arm == "T4":
        use, rejected = _verify_peer_records_content(peers, pc)
        tamper_alerts = len(rejected)
        for p in rejected:
            r = _recover_peer_from_chain(p, pc)
            if r is not None:
                use = use + [r]
                recovered += 1
    else:
        raise ValueError(arm)

    res = evidence_vote(scenario, use)
    res.update({
        "classification":       "valid",
        "trace_rate":           1.0 if arm in ("T3", "T4", "T5", "T5x") and use else 0.0,
        "latency_ms":           0.0,
        "latency_retrieval_ms": 0.0,
        "latency_inference_ms": 0.0,
        "model_used":           "evidence_vote",
        "tamper_alerts":        tamper_alerts,
        "peers_recovered":      recovered,
        "condition":            arm,
        "llm_output":           "",
        "violations":           [],
    })
    return res
