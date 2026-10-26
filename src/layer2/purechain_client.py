"""
purechain_client.py
Layer 2 — web3.py interface to PureChain.

Wraps the SupplyEventLogger smart contract with typed Python methods
for logging Type-A (asset state) and Type-B (routing condition)
events, logging decisions, and verifying on-chain records.

Works with both:
  - Local Hardhat node (development, localhost:8545)
  - Real PureChain PoA² node (final experiments, localhost:8546)

The active network is controlled by .env (PURECHAIN_RPC_URL).

Gas price note:
  Hardhat (EIP-1559): gas_price = current baseFeePerGas + priority fee
  PureChain PoA²:     gas_price = 0 (gas-free consensus)
  Both are handled by querying self.w3.eth.gas_price at transaction time.
"""

from __future__ import annotations
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger

from web3 import Web3
from web3.exceptions import TransactionNotFound
from web3.contract import Contract
from web3.types import TxReceipt
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.layer1.event_generator import AssetStateRecord, RoutingConditionRecord

load_dotenv()

ABI_PATH = Path(__file__).parent.parent.parent / \
           "artifacts/contracts/SupplyEventLogger.sol/SupplyEventLogger.json"


# Confirmation depth required before a committed event is accepted.
# This is an OPERATIONAL policy, not a finality proof: PureChain's PoA2
# builds on Clique proof of authority, which allows short forks and has
# no formal finality (EIP-225), and the endpoint does not expose its
# sealer set.  Six blocks bound the reorganisation risk we accept.  Peer
# records are anchored once, long before they are retrieved, so the depth
# costs no availability in this use.
MIN_CONFIRMATIONS_DEFAULT = 6

_HANDLE_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


class VerificationUnavailable(RuntimeError):
    """The verifier could not obtain an answer from the chain.

    Distinct from a verification *failure*: a missing receipt, a wrong
    recipient, or a digest mismatch is evidence against the record,
    whereas a transport error is no evidence at all.  Callers must not
    turn the second into the first.  The decision pipeline fails closed
    on it instead (``llm_agent_v14.decide_v14``).
    """


@dataclass
class VerificationPolicy:
    """
    Checks a thin client must perform before trusting an RPC
    response.

    A client that queries a single JSON-RPC endpoint over HTTPS does
    not independently establish the authenticity or finality of the
    chain state it is shown: a compromised endpoint can return a
    fabricated receipt.  These checks remove the cheap forgeries --
    a transaction that never succeeded, one addressed to an
    attacker-deployed contract exposing the same ABI, one from a
    different chain, one submitted by an unauthorised writer, one that
    is not a peer-anchoring call at all, or one not yet buried under
    enough blocks to be considered final.  They do not remove the
    residual assumption that a queried endpoint is honest; see
    ``verify_content_quorum`` for the multi-endpoint acceptance rule.

    On ``check_writer``: the deployed contract restricts the logging
    functions with ``onlyOwner``, so a successful transaction to the
    expected contract that decodes to ``logAssetState`` can only have
    come from the owner; any other sender reverts with status 0.
    Authorisation is therefore enforced on-chain.  The client checks the
    sender anyway, against a writer address pinned in its own
    configuration (``PURECHAIN_ANCHOR_ADDRESS``), so that the check does
    not rest on an answer from the endpoint it is meant to police.

    ``quorum_rule``: "unanimous" accepts only when every configured
    endpoint is reachable and returns the same accepting answer, so one
    honest endpoint suffices to block a fabricated answer.  "m_of_m"
    accepts on ``quorum_min_agree`` agreeing endpoints and needs the
    stronger assumption that fewer than that many collude.
    """
    check_chain_id:    bool = True
    check_status:      bool = True
    check_recipient:   bool = True
    check_writer:      bool = True
    check_function:    bool = True
    min_confirmations: int  = MIN_CONFIRMATIONS_DEFAULT
    expected_chain_id: Optional[int] = None
    expected_writer:   Optional[str] = None
    quorum_rule:       str  = "unanimous"
    quorum_min_agree:  int  = 2


class PureChainClient:
    """
    Interface to the SupplyEventLogger smart contract on PureChain.

    Provides:
      - log_asset_state()       : write Type-A record on-chain
      - log_routing_condition() : write Type-B record on-chain
      - log_decision()          : write LLM decision audit on-chain
      - verify_event()          : check that an event ID exists on-chain
      - verify_tx_hash()        : check that a tx hash exists on-chain
    """

    def __init__(
        self,
        rpc_url:          Optional[str] = None,
        private_key:      Optional[str] = None,
        contract_address: Optional[str] = None,
        policy:           Optional["VerificationPolicy"] = None,
        quorum_urls:      Optional[List[str]] = None,
    ) -> None:
        self.rpc_url          = rpc_url or os.getenv("PURECHAIN_RPC_URL", "http://127.0.0.1:8545")
        self.private_key      = private_key or os.getenv("PURECHAIN_PRIVATE_KEY")
        self.contract_address = contract_address or os.getenv("SUPPLY_EVENT_LOGGER_ADDRESS")

        env_chain = os.getenv("PURECHAIN_CHAIN_ID")
        self.expected_chain_id = int(env_chain) if env_chain else None
        self.policy = policy or VerificationPolicy(
            expected_chain_id=self.expected_chain_id)
        if self.policy.expected_chain_id is None:
            self.policy.expected_chain_id = self.expected_chain_id

        # The identity that anchored the corpus.  Configurable, because a
        # verifier is not in general the writer and must be able to name
        # the authorised writer without holding its key.
        env_writer = os.getenv("PURECHAIN_ANCHOR_ADDRESS")
        if self.policy.expected_writer is None and env_writer:
            self.policy.expected_writer = env_writer

        # Additional endpoints for N-of-M agreement.  Empty in the
        # current deployment: PureChain exposes one public RPC, so
        # single-endpoint honesty remains an explicit assumption.
        env_quorum = os.getenv("PURECHAIN_QUORUM_URLS", "")
        self.quorum_urls = quorum_urls if quorum_urls is not None else [
            u.strip() for u in env_quorum.split(",") if u.strip()]

        # Tally of endpoint-check outcomes, for reporting.
        self.check_stats: Counter = Counter()

        self.w3:       Optional[Web3]     = None
        self.contract: Optional[Contract] = None
        self.account:  Optional[str]      = None
        self._connected = False
        self.writer_source = "unset"
        self._head: Optional[Tuple[int, float]] = None
        self._quorum_clients: Dict[str, "PureChainClient"] = {}

    # ── Connection ─────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

            if not self.w3.is_connected():
                logger.error(
                    f"Cannot connect to node: {self.rpc_url}. "
                    "Start Hardhat: npx hardhat node"
                )
                return False

            chain_id = self.w3.eth.chain_id
            logger.info(
                f"Connected to node: {self.rpc_url} "
                f"(chain {chain_id})"
            )

            if (self.policy.check_chain_id
                    and self.policy.expected_chain_id is not None
                    and chain_id != self.policy.expected_chain_id):
                logger.error(
                    f"Chain ID mismatch: endpoint reports {chain_id}, "
                    f"expected {self.policy.expected_chain_id}.  "
                    "Refusing to treat this endpoint as authoritative."
                )
                return False

            if self.private_key:
                account = self.w3.eth.account.from_key(self.private_key)
                self.account = account.address
            else:
                # Read-only verifier: verification needs no account at all,
                # and a public node need not answer eth_accounts.
                self.account = None
                logger.info("No private key set: read-only verifier "
                            "(logging methods are unavailable).")

            logger.info(f"Account: {self.account}")

            if self.contract_address and ABI_PATH.exists():
                self._load_contract()
                # The authorised writer is defined by the contract, not by
                # local configuration: the logging entry points are
                # onlyOwner, so the owner is exactly the identity whose
                # events can exist.  Reading it from the contract means a
                # verifier that holds no key can still name the writer it
                # expects.  An explicit PURECHAIN_ANCHOR_ADDRESS overrides
                # it, for a verifier that prefers not to take even this
                # from the endpoint.
                try:
                    owner = self.contract.functions.owner().call()
                except Exception as exc:
                    owner = None
                    logger.warning(f"Could not read the contract owner: {exc}")
                if self.policy.check_writer:
                    if self.policy.expected_writer:
                        self.writer_source = "pinned"
                        if owner and (Web3.to_checksum_address(owner)
                                      != Web3.to_checksum_address(
                                          self.policy.expected_writer)):
                            logger.error(
                                "Pinned anchoring writer "
                                f"{self.policy.expected_writer} differs from "
                                f"the owner the endpoint reports ({owner}); "
                                "every handle will fail the writer check.")
                    elif owner:
                        # Taking the writer from the endpoint under test is
                        # weaker than pinning it; say so.
                        self.policy.expected_writer = owner
                        self.writer_source = "endpoint_owner"
                        logger.warning(
                            "PURECHAIN_ANCHOR_ADDRESS not set; taking the "
                            f"authorised writer from the endpoint ({owner}).")
                    else:
                        logger.warning("Writer check inactive: no pinned "
                                       "writer and no readable owner.")
            else:
                logger.warning(
                    "Contract not loaded. "
                    "Deploy: npx hardhat run scripts/deploy.js --network localhost"
                )

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _load_contract(self) -> None:
        with open(ABI_PATH) as f:
            artifact = json.load(f)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=artifact["abi"],
        )
        logger.info(f"Contract loaded: {self.contract_address}")

    def _require_connected(self) -> None:
        if not self._connected or self.w3 is None:
            raise RuntimeError("Not connected. Call client.connect() first.")

    def _require_contract(self) -> None:
        self._require_connected()
        if self.contract is None:
            raise RuntimeError(
                "Contract not loaded. "
                "Deploy SupplyEventLogger and set SUPPLY_EVENT_LOGGER_ADDRESS in .env"
            )

    # ── Logging methods ────────────────────────────────────────

    def log_asset_state(self, record: AssetStateRecord) -> Tuple[str, str]:
        """
        Write a Type-A asset state record to PureChain.
        Returns (tx_hash, event_id_hex).
        """
        self._require_contract()

        soc_bp  = int(record.soc_pct * 100)
        temp_bp = int(record.temperature_c * 100)
        deg_bp  = int(record.degradation_pct * 100)

        tx = self.contract.functions.logAssetState(
            record.asset_id,
            record.hub_location,
            soc_bp,
            temp_bp,
            deg_bp,
            bytes.fromhex(record.sensor_hash[:64]),
        ).build_transaction({
            "from":     self.account,
            "nonce":    self.w3.eth.get_transaction_count(self.account, "pending"),
            "gas":      500_000,
            "gasPrice": 0,  # PureChain PoA² is gas-free
        })

        receipt = self._send(tx)
        tx_hash  = receipt["transactionHash"].hex()
        event_id = self._extract_event_id(receipt, "AssetStateLogged")
        record.chain_tx_hash = tx_hash

        logger.info(f"Type-A on-chain: {record.asset_id}  tx={tx_hash[:16]}...")
        return tx_hash, event_id

    def log_routing_condition(self, record: RoutingConditionRecord) -> Tuple[str, str]:
        """
        Write a Type-B routing condition record to PureChain.
        Returns (tx_hash, event_id_hex).
        """
        self._require_contract()

        tx = self.contract.functions.logRoutingCondition(
            record.path_id,
            record.hop_origin,
            record.hop_dest,
            int(record.transit_h),
            int(record.amb_temp_min_c * 100),
            int(record.amb_temp_max_c * 100),
            record.temp_controlled,
            record.carrier_id,
            int(record.carrier_variance_c * 100),
        ).build_transaction({
            "from":     self.account,
            "nonce":    self.w3.eth.get_transaction_count(self.account, "pending"),
            "gas":      500_000,
            "gasPrice": 0,  # PureChain PoA² is gas-free
        })

        receipt = self._send(tx)
        tx_hash  = receipt["transactionHash"].hex()
        event_id = self._extract_event_id(receipt, "RoutingConditionLogged")
        record.chain_tx_hash = tx_hash

        logger.info(
            f"Type-B on-chain: {record.path_id} "
            f"{record.hop_origin}->{record.hop_dest}  tx={tx_hash[:16]}..."
        )
        return tx_hash, event_id

    def log_decision(
        self,
        classification: str,
        decision_type:  str,
        trace_rate:     float,
    ) -> str:
        """Write an LLM decision audit entry to PureChain. Returns tx_hash."""
        self._require_contract()

        tx = self.contract.functions.logDecision(
            classification,
            decision_type,
            int(trace_rate * 10_000),
        ).build_transaction({
            "from":     self.account,
            "nonce":    self.w3.eth.get_transaction_count(self.account, "pending"),
            "gas":      300_000,
            "gasPrice": 0,  # PureChain PoA² is gas-free
        })

        receipt = self._send(tx)
        tx_hash  = receipt["transactionHash"].hex()

        logger.info(
            f"Decision on-chain: [{classification}] {decision_type} "
            f"trace={trace_rate:.2%}  tx={tx_hash[:16]}..."
        )
        return tx_hash

    # ── v14 peer anchoring (content-bound verification, D1) ───

    V14_PEER_MARKER = "V14PEER"

    def log_v14_peer(
        self,
        payload_json: str,
        digest_hex:   str,
        soc_bp:       int = 0,
        temp_c:       int = 0,
        stage:        int = 0,
    ) -> str:
        """
        Anchor a v14 peer record on PureChain: the full canonical
        payload goes in the assetId string field (enabling chain-based
        recovery) and its SHA-256 content digest in the bytes32
        sensorHash field (enabling content-bound verification).
        Returns the transaction hash.
        """
        self._require_contract()
        tx = self.contract.functions.logAssetState(
            payload_json,
            self.V14_PEER_MARKER,
            int(soc_bp),
            int(temp_c),
            int(stage),
            bytes.fromhex(digest_hex.replace("0x", "")[:64]),
        ).build_transaction({
            "from":     self.account,
            "nonce":    self.w3.eth.get_transaction_count(self.account, "pending"),
            "gas":      1_000_000,
            "gasPrice": 0,  # PureChain PoA² is gas-free
        })
        receipt = self._send(tx)
        return receipt["transactionHash"].hex()

    def verification_available(self) -> bool:
        """True when the verifier can obtain answers from the chain."""
        return bool(self._connected and self.w3 is not None)

    def _rpc(self, fn, *args):
        """Run one RPC call; map transport failure to VerificationUnavailable."""
        try:
            return fn(*args)
        except TransactionNotFound:
            raise
        except Exception as exc:
            raise VerificationUnavailable(
                f"{type(exc).__name__}: {exc}") from exc

    def _head_block(self) -> int:
        now = time.monotonic()
        if self._head is None or now - self._head[1] > 5.0:
            self._head = (int(self._rpc(lambda: self.w3.eth.block_number)), now)
        return self._head[0]

    def _endpoint_checks_ex(
        self, tx_hash: str, need_tx: bool = False,
    ) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
        """
        Validate the context of a transaction before any of its content
        is trusted.  Returns (ok, failed_check_names, decoded) where
        ``decoded`` = {"func": name, "args": {...}} when the calldata was
        decoded, else None.

        Checks, in order: the handle is well formed; the receipt exists;
        the transaction succeeded; it was addressed to the expected
        contract; it was sent by the pinned anchoring identity; it is
        buried under at least ``min_confirmations`` blocks; and its
        calldata decodes to ``logAssetState`` carrying the peer-event
        marker.  The chain identifier is checked once, at connect time.

        Raises ``VerificationUnavailable`` on a transport error, which
        is not evidence against the record.
        """
        if not self.verification_available():
            raise VerificationUnavailable("client not connected")
        if not tx_hash:
            self.check_stats["empty_handle"] += 1
            return False, ["empty_handle"], None
        if not _HANDLE_RE.match(tx_hash):
            self.check_stats["malformed_handle"] += 1
            return False, ["malformed_handle"], None

        try:
            receipt = self._rpc(self.w3.eth.get_transaction_receipt, tx_hash)
        except TransactionNotFound:
            receipt = None
        if receipt is None:
            self.check_stats["receipt_missing"] += 1
            return False, ["receipt_missing"], None

        fails: List[str] = []

        if self.policy.check_status and receipt.get("status") != 1:
            fails.append("tx_reverted")

        if self.policy.check_recipient:
            to = receipt.get("to")
            if not to or not self.contract_address:
                fails.append("recipient_unknown")
            else:
                try:
                    same = (Web3.to_checksum_address(to)
                            == Web3.to_checksum_address(self.contract_address))
                except Exception:
                    same = False
                if not same:
                    fails.append("wrong_recipient")

        if self.policy.check_writer and self.policy.expected_writer:
            frm = receipt.get("from")
            if not frm:
                fails.append("writer_unknown")
            else:
                try:
                    same_w = (Web3.to_checksum_address(frm)
                              == Web3.to_checksum_address(
                                  self.policy.expected_writer))
                except Exception:
                    same_w = False
                if not same_w:
                    fails.append("unauthorised_writer")

        if self.policy.min_confirmations > 0:
            depth = self._head_block() - int(receipt["blockNumber"]) + 1
            if depth < self.policy.min_confirmations:
                fails.append("insufficient_finality")

        decoded = None
        if not fails and (self.policy.check_function or need_tx):
            if self.contract is None:
                fails.append("function_unknown")
            else:
                try:
                    tx = self._rpc(self.w3.eth.get_transaction, tx_hash)
                except TransactionNotFound:
                    tx = None
                if tx is None:
                    fails.append("receipt_missing")
                else:
                    try:
                        func, args = self.contract.decode_function_input(
                            tx["input"])
                        decoded = {"func": func.fn_name, "args": dict(args)}
                    except Exception:
                        decoded = None
                    if decoded is None:
                        fails.append("function_undecodable")
                    elif self.policy.check_function and (
                            decoded["func"] != "logAssetState"
                            or decoded["args"].get("hubLocation")
                            != self.V14_PEER_MARKER):
                        fails.append("not_peer_event")

        self.check_stats["checked"] += 1
        for f in fails:
            self.check_stats[f] += 1
        if not fails:
            self.check_stats["passed"] += 1
        return (not fails), fails, decoded

    def endpoint_checks(self, tx_hash: str) -> Tuple[bool, List[str]]:
        """(ok, failed_check_names); see ``_endpoint_checks_ex``."""
        ok, fails, _ = self._endpoint_checks_ex(tx_hash)
        return ok, fails

    def get_peer_payload(self, tx_hash: str) -> Optional[Tuple[str, str]]:
        """
        Fetch a committed v14 peer payload by decoding the transaction's
        calldata.  Returns (payload_json, digest_hex), or None if the
        transaction is missing, fails the endpoint checks, or is not a
        v14 peer event.  Raises ``VerificationUnavailable`` on transport
        failure.
        """
        if not self.verification_available():
            raise VerificationUnavailable('client not connected')
        self._require_contract()
        ok, _, decoded = self._endpoint_checks_ex(tx_hash, need_tx=True)
        if not ok or decoded is None:
            return None
        args = decoded["args"]
        if decoded["func"] != "logAssetState":
            return None
        if args.get("hubLocation") != self.V14_PEER_MARKER:
            return None
        digest = args["sensorHash"].hex().replace("0x", "").lower()
        return args["assetId"], digest

    # ── multi-endpoint acceptance ─────────────────────────────

    def _quorum_endpoints(self) -> List[str]:
        return [self.rpc_url] + [u for u in self.quorum_urls
                                 if u != self.rpc_url]

    def _peer_client(self, url: str) -> "PureChainClient":
        """A read-only client for another endpoint, sharing this policy
        (including the pinned writer and chain identifier)."""
        c = self._quorum_clients.get(url)
        if c is None:
            c = PureChainClient(rpc_url=url, private_key=None,
                                contract_address=self.contract_address,
                                policy=self.policy, quorum_urls=[])
            c.connect()
            self._quorum_clients[url] = c
        return c

    def _payloads_from_all(self, tx_hash: str):
        """Ask every configured endpoint.  Returns (answers, failed_urls),
        where answers maps url -> (payload, digest) or None."""
        answers: Dict[str, Optional[Tuple[str, str]]] = {}
        failed: List[str] = []
        for url in self._quorum_endpoints():
            client = self if url == self.rpc_url else self._peer_client(url)
            try:
                answers[url] = client.get_peer_payload(tx_hash)
            except VerificationUnavailable:
                failed.append(url)
        return answers, failed

    def verify_content_quorum(
        self, tx_hash: str, local_digest_hex: str,
        rule: Optional[str] = None, min_agree: Optional[int] = None,
    ) -> Tuple[bool, int, int, int, List[str]]:
        """
        Content-bound verification across every configured endpoint.
        Returns (accepted, n_agree, n_reachable, n_configured, failed_urls).

        Under the default unanimity rule a record is accepted only if
        every configured endpoint is reachable and each reports a
        committed payload whose digest equals the locally recomputed
        one; any unreachable or disagreeing endpoint rejects.  One honest
        endpoint therefore suffices to block a fabricated answer, which
        is the rule the single-honest-endpoint assumption supports.
        With no additional endpoints configured this is the
        single-endpoint case, reported as such.
        """
        rule = rule or self.policy.quorum_rule
        need = min_agree or self.policy.quorum_min_agree
        want = (local_digest_hex or "").replace("0x", "").lower()
        answers, failed = self._payloads_from_all(tx_hash)
        n_conf = len(answers) + len(failed)
        n_reach = len(answers)
        n_agree = sum(1 for a in answers.values()
                      if a is not None and a[1] == want)
        if rule == "unanimous":
            accepted = (n_reach == n_conf and n_agree == n_conf and n_conf > 0)
        elif rule == "m_of_m":
            accepted = n_agree >= min(need, n_conf) and n_agree > 0
        else:
            raise ValueError(f"unknown quorum rule {rule!r}")
        if n_reach == 0:
            raise VerificationUnavailable("no endpoint reachable")
        return accepted, n_agree, n_reach, n_conf, failed

    def get_peer_payload_quorum(self, tx_hash: str) -> Optional[Tuple[str, str]]:
        """Recovery read under unanimity: the payload is returned only if
        every configured endpoint is reachable and returns the identical
        (payload, digest) pair."""
        answers, failed = self._payloads_from_all(tx_hash)
        if not answers:
            raise VerificationUnavailable("no endpoint reachable")
        if failed:
            return None
        vals = list(answers.values())
        if any(v is None for v in vals) or len(set(vals)) != 1:
            return None
        return vals[0]

    def verify_content(self, tx_hash: str, local_digest_hex: str) -> bool:
        """
        Content-bound verification predicate: the handle must resolve
        to a committed v14 peer event whose on-chain content digest
        equals the digest recomputed from the locally retrieved record.
        Uses the multi-endpoint rule when extra endpoints are configured.
        """
        if not tx_hash or not local_digest_hex:
            return False
        if self.quorum_urls:
            return self.verify_content_quorum(tx_hash, local_digest_hex)[0]
        got = self.get_peer_payload(tx_hash)
        if got is None:
            return False
        return got[1] == local_digest_hex.replace("0x", "").lower()

    def recover_payload(self, tx_hash: str) -> Optional[Tuple[str, str]]:
        """Payload read used by recovery; multi-endpoint when configured."""
        if self.quorum_urls:
            return self.get_peer_payload_quorum(tx_hash)
        return self.get_peer_payload(tx_hash)

    # ── Verification ───────────────────────────────────────────

    def verify_event(self, event_id_hex: str) -> Tuple[bool, int]:
        """Check that an event ID exists on-chain via the contract mapping."""
        self._require_contract()
        event_id_bytes = bytes.fromhex(event_id_hex.replace("0x", ""))
        exists, block_number = self.contract.functions.verifyEvent(
            event_id_bytes
        ).call()
        return exists, int(block_number)

    def verify_tx_hash(self, tx_hash: str) -> bool:
        """
        Handle-existence verification: the handle must resolve to a
        final, successful peer-anchoring transaction sent by the
        authorised writer to the expected contract on the expected
        chain.

        Note what this does NOT establish: nothing binds the handle to
        the CONTENT of the record that carried it, so a record whose
        fields were altered while retaining a genuine handle still
        passes.  ``verify_content`` closes that gap.
        """
        ok, _ = self.endpoint_checks(tx_hash)
        return ok

    # ── Internal helpers ───────────────────────────────────────

    def _send(self, tx: Dict[str, Any]) -> TxReceipt:
        """Sign and broadcast a transaction, return the receipt."""
        if self.private_key:
            signed  = self.w3.eth.account.sign_transaction(tx, self.private_key)
            raw     = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        else:
            tx.pop("gasPrice", None)
            tx_hash = self.w3.eth.send_transaction(tx)

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        if receipt["status"] != 1:
            raise RuntimeError(f"Transaction reverted: {tx_hash.hex()}")
        return receipt

    def _extract_event_id(self, receipt: TxReceipt, event_name: str) -> str:
        """Extract the eventId bytes32 from transaction logs."""
        try:
            event = getattr(self.contract.events, event_name)()
            logs  = event.process_receipt(receipt)
            if logs:
                return logs[0]["args"]["eventId"].hex()
        except Exception as e:
            logger.warning(f"Could not parse event logs ({event_name}): {e}")
        return ""

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Mock client ────────────────────────────────────────────────

class MockPureChainClient:
    """
    Drop-in replacement for PureChainClient when no blockchain node
    is available (unit tests, CI). All methods return plausible values.
    """

    def connect(self) -> bool:
        logger.warning("MockPureChainClient active — no real blockchain")
        return True

    def log_asset_state(self, record: AssetStateRecord) -> Tuple[str, str]:
        import hashlib, time
        tx = hashlib.sha256(f"{record.asset_id}{time.time()}".encode()).hexdigest()
        record.chain_tx_hash = tx
        return tx, tx[:64]

    def log_routing_condition(self, record: RoutingConditionRecord) -> Tuple[str, str]:
        import hashlib, time
        tx = hashlib.sha256(f"{record.path_id}{time.time()}".encode()).hexdigest()
        record.chain_tx_hash = tx
        return tx, tx[:64]

    def log_decision(self, classification: str, decision_type: str, trace_rate: float) -> str:
        import hashlib, time
        return hashlib.sha256(f"{classification}{time.time()}".encode()).hexdigest()

    def verify_tx_hash(self, tx_hash: str) -> bool:
        return bool(tx_hash) and len(tx_hash) == 64

    def verification_available(self) -> bool:
        # The mock accepts any 64-character handle, so it must never
        # stand in for a verifier in C3/C4.
        return False

    def recover_payload(self, tx_hash: str):
        return None

    def log_v14_peer(self, payload_json: str, digest_hex: str,
                     soc_bp: int = 0, temp_c: int = 0, stage: int = 0) -> str:
        import hashlib, time
        return hashlib.sha256(f"{digest_hex}{time.time()}".encode()).hexdigest()

    def get_peer_payload(self, tx_hash: str):
        return None  # mock has no chain to fetch from

    def verify_content(self, tx_hash: str, local_digest_hex: str) -> bool:
        return False  # fail closed without a real chain

    def verify_event(self, event_id_hex: str) -> Tuple[bool, int]:
        return True, 1

    @property
    def is_connected(self) -> bool:
        return True


# ── Factory ────────────────────────────────────────────────────

def get_client(mock: bool = False) -> PureChainClient | MockPureChainClient:
    """
    Return real or mock client.
    Real client is used when PURECHAIN_RPC_URL is set in .env and mock=False.
    """
    if mock or not os.getenv("PURECHAIN_RPC_URL"):
        return MockPureChainClient()
    client = PureChainClient()
    client.connect()
    return client


if __name__ == "__main__":
    client = get_client()
    print(f"Connected: {client.is_connected}")