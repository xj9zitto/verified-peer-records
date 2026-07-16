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
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from loguru import logger

from web3 import Web3
from web3.contract import Contract
from web3.types import TxReceipt
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.layer1.event_generator import AssetStateRecord, RoutingConditionRecord

load_dotenv()

ABI_PATH = Path(__file__).parent.parent.parent / \
           "artifacts/contracts/SupplyEventLogger.sol/SupplyEventLogger.json"


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
    ) -> None:
        self.rpc_url          = rpc_url or os.getenv("PURECHAIN_RPC_URL", "http://127.0.0.1:8545")
        self.private_key      = private_key or os.getenv("PURECHAIN_PRIVATE_KEY")
        self.contract_address = contract_address or os.getenv("SUPPLY_EVENT_LOGGER_ADDRESS")

        self.w3:       Optional[Web3]     = None
        self.contract: Optional[Contract] = None
        self.account:  Optional[str]      = None
        self._connected = False

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

            logger.info(
                f"Connected to node: {self.rpc_url} "
                f"(chain {self.w3.eth.chain_id})"
            )

            if self.private_key:
                account = self.w3.eth.account.from_key(self.private_key)
                self.account = account.address
            else:
                self.account = self.w3.eth.accounts[0]
                logger.warning(
                    "No private key set — using default Hardhat account. "
                    "Set PURECHAIN_PRIVATE_KEY in .env for production."
                )

            logger.info(f"Account: {self.account}")

            if self.contract_address and ABI_PATH.exists():
                self._load_contract()
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

    def get_peer_payload(self, tx_hash: str) -> Optional[Tuple[str, str]]:
        """
        Fetch a committed v14 peer payload from the chain by decoding
        the transaction's calldata.  Returns (payload_json, digest_hex)
        or None if the transaction is missing or is not a v14 peer event.
        """
        self._require_contract()
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            func, args = self.contract.decode_function_input(tx["input"])
            if func.fn_name != "logAssetState":
                return None
            if args.get("hubLocation") != self.V14_PEER_MARKER:
                return None
            digest = args["sensorHash"].hex().replace("0x", "").lower()
            return args["assetId"], digest
        except Exception:
            return None

    def verify_content(self, tx_hash: str, local_digest_hex: str) -> bool:
        """
        Content-bound verification predicate: the handle must resolve
        to a committed v14 peer event whose on-chain content digest
        equals the digest of the locally retrieved record.
        """
        if not tx_hash or not local_digest_hex:
            return False
        got = self.get_peer_payload(tx_hash)
        if got is None:
            return False
        return got[1] == local_digest_hex.replace("0x", "").lower()

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
        Confirm a transaction hash exists on-chain and was successful.
        Used by VectorIndex for claim-level traceability in C3 mode.
        """
        self._require_connected()
        if not tx_hash:
            return False
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            return receipt is not None and receipt["status"] == 1
        except Exception:
            return False

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