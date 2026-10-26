"""
v14_boundary_attack.py — attacks that lie outside the content-integrity
guarantee.

The attacks in ``v14_attack.py`` rewrite a retrieved peer's outcome
fields.  That changes the record's canonical payload, so its digest no
longer matches the committed one and content-bound verification rejects
it.  Reviewer 2 observed that this exercises only the part of the attack
surface the guarantee covers, and asked for the complement: an attack
that substitutes complete genuine record-handle pairs, and one that
withholds records.  Section IV-D already excludes selection integrity
and completeness from what the predicates claim; these two attacks turn
that exclusion from an assertion into a measurement.

  genuine_substitution  Every retrieved peer is replaced by a DIFFERENT
                        but genuine peer, carried with its own genuine
                        handle, chosen for a benign realized outcome.
                        Both record and handle are authentic, so the
                        digest comparison succeeds and chain recovery,
                        keyed on the record's own identity fields,
                        returns the substituted peer rather than the
                        intended one.  Nothing in the mechanism fires.

  withhold              The peers whose realized outcome is severe are
                        dropped.  Every survivor verifies.  Detecting
                        this needs a known expected set, which the
                        predicates do not have.

Both attacks also apply the rosy observable-window poisoning of
``v14_attack.poison_obs_summary``, so they differ from the in-scope
attacks only in how the retrieved evidence is manipulated.

The draw is deterministic in (seed, scenario id) exactly as in
``v14_attack``, so every arm faces identical attack instances and the
results are comparable pair for pair.
"""

from __future__ import annotations
import copy
import hashlib
from typing import Any, Dict, List

from src.dataset.v14_attack import ROSY_SNAPSHOT, poison_obs_summary
from src.layer2.v14_content import peer_key

BOUNDARY_ATTACKS = ("genuine_substitution", "withhold")


def _u(*parts: Any) -> float:
    """Deterministic uniform draw in [0, 1) from the given parts."""
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def is_severe(peer: Dict[str, Any]) -> bool:
    """Whether a peer's realized outcome is evidence of escalation.

    These are the records that push the peer vote toward ``replace`` or
    ``urgent``, and therefore the ones an attacker wants out of the
    context window.
    """
    out = peer.get("outcome") or {}
    try:
        stage = int(out.get("failure_stage_max", 0))
    except (TypeError, ValueError):
        stage = 0
    return stage >= 2 or bool(out.get("did_escalate", False))


def benign_peer_pool(scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Distinct genuine peers with a benign realized outcome.

    These are real, anchored records: substituting one preserves both
    the digest match and the payload-to-identity consistency that
    recovery checks, which is what makes the attack invisible to the
    mechanism.  Sorted by identity key so the pool is order-stable
    across processes.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for sc in scenarios:
        for p in sc.get("asset_records", []) or []:
            if is_severe(p):
                continue
            out = p.get("outcome") or {}
            if int(out.get("failure_stage_max", 0)) != 0:
                continue
            seen.setdefault(peer_key(p), p)
    return [seen[k] for k in sorted(seen)]


def apply_genuine_substitution(
    scenario: Dict[str, Any],
    pool:     List[Dict[str, Any]],
    seed:     int,
) -> Dict[str, Any]:
    """Swap every retrieved peer for a genuine benign peer and its handle."""
    s = copy.deepcopy(scenario)
    sid = s.get("scenario_id", "")

    if "obs_window" in s and "summary" in s["obs_window"]:
        poison_obs_summary(s["obs_window"]["summary"])
    if "obs_window" in s and "snapshot" in s["obs_window"]:
        s["obs_window"]["snapshot"] = dict(ROSY_SNAPSHOT)

    peers = s.get("asset_records", []) or []
    if pool and peers:
        # Draw without replacement so the substituted set has the same
        # cardinality as the real one: only which peers changed, not how
        # many, so the comparison with the in-scope attacks is clean.
        order = sorted(range(len(pool)),
                       key=lambda i: _u(seed, "subst", sid, i))
        chosen = [pool[order[i % len(order)]] for i in range(len(peers))]
        s["asset_records"] = [copy.deepcopy(c) for c in chosen]

    s["_poisoned"] = True
    s["_attack_type"] = "genuine_substitution"
    return s


def apply_withhold(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the peers whose realized outcome is severe."""
    s = copy.deepcopy(scenario)

    if "obs_window" in s and "summary" in s["obs_window"]:
        poison_obs_summary(s["obs_window"]["summary"])
    if "obs_window" in s and "snapshot" in s["obs_window"]:
        s["obs_window"]["snapshot"] = dict(ROSY_SNAPSHOT)

    peers = s.get("asset_records", []) or []
    s["asset_records"] = [p for p in peers if not is_severe(p)]

    s["_poisoned"] = True
    s["_attack_type"] = "withhold"
    s["_n_withheld"] = len(peers) - len(s["asset_records"])
    return s
