"""
rag_retrieval.py
Layer 2 — RAG retrieval interface with condition-aware routing.

Four experimental conditions:
  C0 — Rule-engine baseline: no LLM, no retrieval
  C1 — Ungrounded LLM: parametric memory only, no retrieval
  C2 — LLM + SQL RAG: retrieves without hash verification;
       15% of records are flagged as corrupted (simulating real-world
       data quality errors in SQL supply chain databases)
  C3 — LLM + PureChain RAG (proposed): every retrieved record is
       verified against the on-chain tx hash; tampered or missing
       records are excluded before the LLM sees the context

Hash verification in C3 calls verify_tx_hash() on the real
PureChainClient, which checks the Hardhat (or PoA² node) directly.
No mock seeding is required.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.layer2.vector_index import VectorIndex
from src.layer2.purechain_client import PureChainClient, get_client
from src.config import K_ASSET, K_ROUTING

C2_CORRUPTION_RATE = 0.15
TX_REGISTRY        = Path("data/synthetic/tx_registry.json")


class RAGRetriever:
    """
    Condition-aware RAG retrieval interface for the 4 experimental
    conditions. Instantiate once per experiment session.
    """

    def __init__(
        self,
        index:           Optional[VectorIndex]     = None,
        client:          Optional[PureChainClient] = None,
        seed:            int   = 42,
        corruption_rate: float | None = None,
    ) -> None:
        self.rng    = random.Random(seed)
        # Corruption rate for C2 — defaults to module-level constant.
        # Pass a different value for the severity sweep ablation.
        self._corruption_rate = corruption_rate if corruption_rate is not None \
                                else C2_CORRUPTION_RATE

        # Real PureChain client — reads PURECHAIN_RPC_URL from .env
        self.client = client or get_client()

        if not self.client.is_connected:
            logger.warning(
                "PureChain node not reachable — C3 verification will fail. "
                "Start Hardhat: npx hardhat node"
            )

        self.index = index or VectorIndex(client=self.client)

        # Build C2 corruption set once at init
        self._corrupt_set: set = self._build_corruption_set()
        logger.info(
            f"RAGRetriever ready — "
            f"corrupt_set={len(self._corrupt_set)} hashes "
            f"({self._corruption_rate:.0%} of {self._total_hashes()} total)"
        )

    # ── Public interface ───────────────────────────────────────

    def retrieve(
        self,
        query:     str,
        condition: str,
        k_asset:   int = K_ASSET,
        k_routing: int = K_ROUTING,
    ) -> Dict[str, Any]:
        """
        Retrieve supply records for a query under an experimental condition.

        Parameters
        ----------
        query     : natural language decision query
        condition : "C0" | "C1" | "C2" | "C3"
        k_asset   : max Type-A records to retrieve
        k_routing : max Type-B records to retrieve

        Returns
        -------
        dict with keys: asset_records, routing_records,
                        prompt_context, tamper_alerts, condition
        """
        if condition in ("C0", "C1"):
            logger.debug(f"[{condition}] Skipping retrieval — empty context")
            return self._empty(condition)

        if condition == "C2":
            return self._retrieve_c2(query, k_asset, k_routing)

        if condition == "C3":
            return self._retrieve_c3(query, k_asset, k_routing)

        raise ValueError(f"Unknown condition '{condition}'. Use C0/C1/C2/C3.")

    # ── Condition implementations ──────────────────────────────

    def _retrieve_c2(
        self, query: str, k_asset: int, k_routing: int
    ) -> Dict[str, Any]:
        """
        C2: Retrieve without on-chain verification.
        Records in the corruption set have their VALUES tampered before
        being passed to the LLM, simulating a SQL supply-chain database
        that has been partially corrupted.  The LLM cannot detect this
        because there is no hash verification — it trusts the data as-is.
        C3 avoids this by excluding any record whose on-chain hash does
        not match, so it never sees corrupted values.
        """
        result = self.index.retrieve(query, k_asset, k_routing, verify=False)

        # Inject corrupted values into flagged records and rebuild
        # prompt_text so the LLM reads the wrong numbers.
        corrupt_count = 0
        for rec in result["asset_records"] + result["routing_records"]:
            if rec.get("chain_tx_hash", "") in self._corrupt_set:
                self._inject_corruption(rec)
                corrupt_count += 1

        # Rebuild prompt_context from the (possibly corrupted) records
        result["prompt_context"] = self.index._build_prompt_context(
            result["asset_records"], result["routing_records"]
        )
        result["tamper_alerts"] = corrupt_count
        result["condition"] = "C2"

        logger.debug(
            f"[C2] asset={len(result['asset_records'])}  "
            f"routing={len(result['routing_records'])}  "
            f"corrupt_injected={corrupt_count}"
        )
        return result

    @staticmethod
    def _inject_corruption(rec: Dict[str, Any]) -> None:
        """
        Mutate a flagged-corrupt record in-place with decision-flipping
        value noise.  The corruption is designed to push each value to
        the opposite side of the relevant physics threshold so the LLM
        is misled into making the wrong supply decision.

        Asset state records (Type-A):
          • SOC: flip across 20% reorder threshold (low→high, high→low)
          • Temperature: subtract 18°C to hide thermal proximity to T_safe
          • Degradation: flip across 80% replacement threshold

        Routing condition records (Type-B):
          • amb_temp_max: subtract 18°C so unsafe routes look thermally safe
          • carrier_variance_c: swap K1↔K3 variance (10↔2) to mislead
            carrier selection
        """
        rtype = rec.get("record_type", "")

        if rtype == "asset_state":
            soc  = float(rec.get("soc_pct", 50))
            temp = float(rec.get("temperature_c", 25))
            deg  = float(rec.get("degradation_pct", 85))

            # Flip SOC across 20% threshold: hide reorder (low→high) or
            # create a false reorder signal (high→low)
            c_soc  = min(95.0, soc  + 50.0) if soc  < 40.0 else max(5.0,  soc  - 50.0)
            # Cool the temperature: make it look thermally safe
            c_temp = max(10.0, temp - 18.0)
            # Flip degradation across 80% threshold: hide replacement need
            c_deg  = min(99.0, deg  + 25.0) if deg  < 70.0 else max(40.0, deg  - 25.0)

            rec["soc_pct"]         = str(round(c_soc,  1))
            rec["temperature_c"]   = str(round(c_temp, 1))
            rec["degradation_pct"] = str(round(c_deg,  1))
            rec["_corrupted"]      = True   # marker for physics-block extractor

            # Regenerate prompt_text with corrupted values
            tx  = rec.get("chain_tx_hash", "")
            aid = rec.get("asset_id", "")
            hub = rec.get("hub_location", "")
            ts  = rec.get("timestamp", "")
            rec["prompt_text"] = (
                f"[SUPPLY RECORD — Asset State]\n"
                f"  Asset ID   : {aid}\n"
                f"  Location   : {hub}\n"
                f"  Timestamp  : {ts}\n"
                f"  SOC        : {c_soc:.1f}%\n"
                f"  Temperature: {c_temp:.1f}°C\n"
                f"  Degradation: {c_deg:.1f}% remaining capacity\n"
                f"  Chain Tx   : {tx}"
            )

        elif rtype == "routing_condition":
            amb_min  = float(rec.get("amb_temp_min_c",     20))
            amb_max  = float(rec.get("amb_temp_max_c",     30))
            variance = float(rec.get("carrier_variance_c",  8))

            # Deflate ambient: make hot routes look thermally safe
            c_min = max(5.0,  amb_min - 18.0)
            c_max = max(10.0, amb_max - 18.0)
            # Swap carrier variance K1↔K3 to maximally mislead selection
            c_var = 2.0 if variance >= 8.0 else 10.0

            rec["amb_temp_min_c"]     = str(round(c_min, 1))
            rec["amb_temp_max_c"]     = str(round(c_max, 1))
            rec["carrier_variance_c"] = str(round(c_var, 1))
            rec["_corrupted"]         = True   # marker for physics-block extractor

            path    = rec.get("path_id",    "?")
            origin  = rec.get("hop_origin", "?")
            dest    = rec.get("hop_dest",   "?")
            transit = float(rec.get("transit_h", 0))
            ctrl    = str(rec.get("temp_controlled", "False")).lower()
            ctrl_s  = "yes" if ctrl in ("true", "1", "yes") else "no"
            carrier = rec.get("carrier_id", "?")
            tx      = rec.get("chain_tx_hash", "")
            rec["prompt_text"] = (
                f"[SUPPLY RECORD — Routing Condition]\n"
                f"  Path       : {path}  ({origin} → {dest})\n"
                f"  Transit    : {transit:.0f} hours\n"
                f"  Ambient    : {c_min:.0f}–{c_max:.0f}°C\n"
                f"  Temp ctrl  : {ctrl_s}\n"
                f"  Carrier    : {carrier}  (±{c_var:.0f}°C variance)\n"
                f"  Chain Tx   : {tx}"
            )

    def _retrieve_c3(
        self, query: str, k_asset: int, k_routing: int
    ) -> Dict[str, Any]:
        """
        C3: Full PureChain hash verification.
        verify_tx_hash() calls the real Hardhat node — records whose
        tx hash does not exist on-chain are excluded from the context.
        """
        result = self.index.retrieve(query, k_asset, k_routing, verify=True)
        result["condition"] = "C3"

        logger.debug(
            f"[C3] asset={len(result['asset_records'])}  "
            f"routing={len(result['routing_records'])}  "
            f"tamper_alerts={result['tamper_alerts']}"
        )
        return result

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _empty(condition: str) -> Dict[str, Any]:
        return {
            "asset_records":   [],
            "routing_records": [],
            "prompt_context":  "",
            "tamper_alerts":   0,
            "condition":       condition,
        }

    def _total_hashes(self) -> int:
        if not TX_REGISTRY.exists():
            return 0
        with open(TX_REGISTRY) as f:
            return len(json.load(f))

    def _build_corruption_set(self) -> set:
        """
        Randomly select 15% of known tx hashes to simulate corrupted
        SQL records in C2. Fixed by seed for reproducibility.
        """
        if not TX_REGISTRY.exists():
            logger.warning(
                "tx_registry.json not found — C2 corruption set is empty. "
                "Run: python scripts/populate_hashes.py"
            )
            return set()
        with open(TX_REGISTRY) as f:
            registry = json.load(f)
        all_hashes = list(registry.keys())
        n_corrupt  = max(1, int(len(all_hashes) * self._corruption_rate))
        return set(self.rng.sample(all_hashes, n_corrupt))


# ── Module-level singleton ─────────────────────────────────────
_retriever: Optional[RAGRetriever] = None


def get_retriever(**kwargs) -> RAGRetriever:
    """Return the module-level RAGRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(**kwargs)
    return _retriever


if __name__ == "__main__":
    """Smoke test — verifies all four conditions return expected shapes."""
    retriever = RAGRetriever()
    query     = "Battery asset B-100 is at H3. Should we reorder?"

    print("\n=== RAG Retrieval Condition Test ===")
    for cond in ["C0", "C1", "C2", "C3"]:
        result = retriever.retrieve(query, cond)
        print(
            f"[{cond}] asset={len(result['asset_records']):>3}  "
            f"routing={len(result['routing_records']):>3}  "
            f"tamper={result['tamper_alerts']}"
        )
        if result["prompt_context"]:
            preview = result["prompt_context"][:80].strip().replace("\n", " ")
            print(f"       {preview}...")