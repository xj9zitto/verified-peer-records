"""
vector_index.py
Layer 2 — ChromaDB vector index and RAG retrieval (Algorithm 1).

Embeds Type-A and Type-B supply records using sentence-transformers
and stores them in two separate ChromaDB collections. Retrieval
returns the top-k most semantically relevant records per type,
with hash verification against PureChain for each retrieved record.

This module implements Algorithm 1 from the paper:
  PureChain-Grounded RAG Retrieval (Condition C3).

Fix (v2):
  Records with an empty chain_tx_hash are not yet logged to the
  blockchain (this is the normal state during dataset generation and
  C0/C1/C2 conditions). An empty hash is treated as "not yet on-chain"
  and skips verification rather than raising a tamper alert.
  Only non-empty hashes that fail pc_client.verify_tx_hash() are
  counted as tamper alerts.
"""

from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import (
    CHROMA_DIR, EMBEDDING_MODEL, K_ASSET, K_ROUTING
)
from src.layer2.serialiser import RecordSerialiser, serialiser
from src.layer2.purechain_client import (
    PureChainClient, MockPureChainClient, get_client
)

COLLECTION_ASSET   = "supply_asset_state"
COLLECTION_ROUTING = "supply_routing_condition"


class VectorIndex:
    """
    Manages two ChromaDB collections — one for Type-A asset state
    records and one for Type-B routing condition records.

    On ingestion, each record is:
      1. Serialised to natural language via RecordSerialiser
      2. Embedded with sentence-transformers
      3. Stored in ChromaDB with the chain_tx_hash as metadata

    On retrieval (Algorithm 1):
      1. Query is embedded
      2. Top-k records are retrieved per collection
      3. Each record's chain_tx_hash is verified against PureChain
         — records with an empty hash skip verification (not yet on-chain)
         — records with a non-empty hash that fails verification raise
           a tamper alert and are excluded from the context
      4. Verified (or unlogged) records are returned as LLM context

    Experimental condition behaviour:
      C0 — rule engine baseline, retrieval not used
      C1 — LLM only, retrieve() returns empty context
      C2 — SQL RAG without hash verification (verify=False)
      C3 — PureChain RAG with full hash verification (verify=True)
    """

    def __init__(
        self,
        client:      Optional[PureChainClient | MockPureChainClient] = None,
        persist_dir: str  = str(CHROMA_DIR),
        mock:        bool = False,
    ) -> None:
        self.serialiser  = serialiser
        self.pc_client   = client or get_client(mock=mock)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self._tx_cache: Dict[str, bool] = {}   # hash → verified result cache

        self.chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.col_asset = self.chroma.get_or_create_collection(
            name=COLLECTION_ASSET,
            metadata={"hnsw:space": "cosine"},
        )
        self.col_routing = self.chroma.get_or_create_collection(
            name=COLLECTION_ROUTING,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"VectorIndex ready — "
            f"asset_docs={self.col_asset.count()} "
            f"routing_docs={self.col_routing.count()} "
            f"embed_model={EMBEDDING_MODEL}"
        )

    # ── Ingestion ──────────────────────────────────────────────

    def add_asset_record(
        self,
        record: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> str:
        """Embed and store a Type-A asset state record."""
        serialised = self.serialiser.serialise_asset(record)
        did = doc_id or self._make_id(
            record.get("asset_id", "?"),
            record.get("timestamp", "?"),
        )
        embedding = self._embed(serialised["embed_text"])
        self.col_asset.add(
            ids=[did],
            embeddings=[embedding],
            documents=[serialised["embed_text"]],
            metadatas=[{
                "chain_tx_hash":   serialised["chain_tx_hash"],
                "record_type":     "asset_state",
                "asset_id":        record.get("asset_id", ""),
                "hub_location":    record.get("hub_location", ""),
                "soc_pct":         str(record.get("soc_pct", 0)),
                "temperature_c":   str(record.get("temperature_c", 0)),
                "degradation_pct": str(record.get("degradation_pct", 100)),
                "prompt_text":     serialised["prompt_text"],
            }],
        )
        logger.debug(f"Asset record indexed: {did}")
        return did

    def add_routing_record(
        self,
        record:  Dict[str, Any],
        doc_id:  Optional[str] = None,
    ) -> str:
        """Embed and store a Type-B routing condition record."""
        serialised = self.serialiser.serialise_routing(record)
        did = doc_id or self._make_id(
            record.get("path_id", "?"),
            record.get("hop_origin", "?"),
            record.get("hop_dest", "?"),
        )
        embedding = self._embed(serialised["embed_text"])
        self.col_routing.add(
            ids=[did],
            embeddings=[embedding],
            documents=[serialised["embed_text"]],
            metadatas=[{
                "chain_tx_hash":      serialised["chain_tx_hash"],
                "record_type":        "routing_condition",
                "path_id":            record.get("path_id", ""),
                "hop_origin":         record.get("hop_origin", ""),
                "hop_dest":           record.get("hop_dest", ""),
                "carrier_id":         record.get("carrier_id", ""),
                "carrier_variance_c": str(record.get("carrier_variance_c", 0)),
                "temp_controlled":    str(record.get("temp_controlled", False)),
                "prompt_text":        serialised["prompt_text"],
            }],
        )
        logger.debug(f"Routing record indexed: {did}")
        return did

    def add_scenario_records(
        self,
        scenario: Dict[str, Any],
    ) -> Tuple[int, int]:
        """
        Index all records from a scenario dict.
        Returns (n_asset_added, n_routing_added).
        """
        na = sum(
            1 for r in scenario.get("asset_records", [])
            if self.add_asset_record(r)
        )
        nr = sum(
            1 for r in scenario.get("routing_records", [])
            if self.add_routing_record(r)
        )
        return na, nr

    # ── Retrieval (Algorithm 1) ────────────────────────────────

    def retrieve(
        self,
        query:     str,
        k_asset:   int  = K_ASSET,
        k_routing: int  = K_ROUTING,
        verify:    bool = True,
    ) -> Dict[str, Any]:
        """
        Algorithm 1 — PureChain-grounded RAG retrieval.

        Hash verification logic:
          - chain_tx_hash == ""  → record not yet on-chain → pass through
                                   (no tamper alert, no exclusion)
          - chain_tx_hash != ""  → verify against PureChain
                                   → fail = tamper alert + excluded from context

        Parameters
        ----------
        query     : natural language decision query
        k_asset   : number of Type-A records to retrieve
        k_routing : number of Type-B records to retrieve
        verify    : if False, all records pass without hash checking (C2 mode)

        Returns
        -------
        dict with keys:
          asset_records   : verified Type-A records
          routing_records : verified Type-B records
          prompt_context  : formatted LLM context string
          tamper_alerts   : count of records that failed verification
        """
        query_embedding = self._embed(query)

        # ── Step 1: retrieve Type-A and Type-B records ─────────
        asset_results = self.col_asset.query(
            query_embeddings=[query_embedding],
            n_results=min(k_asset, max(self.col_asset.count(), 1)),
            include=["metadatas", "documents"],
        )
        routing_results = self.col_routing.query(
            query_embeddings=[query_embedding],
            n_results=min(k_routing, max(self.col_routing.count(), 1)),
            include=["metadatas", "documents"],
        )

        raw_asset   = (asset_results["metadatas"]   or [[]])[0]
        raw_routing = (routing_results["metadatas"] or [[]])[0]

        # ── Step 2: hash verification (parallel for C3) ─────────
        if verify:
            asset_records, routing_records, tamper_alerts = \
                self._verify_parallel(raw_asset, raw_routing)
        else:
            asset_records   = list(raw_asset)
            routing_records = list(raw_routing)
            tamper_alerts   = 0

        # ── Step 3: build prompt context ───────────────────────
        prompt_context = self._build_prompt_context(
            asset_records, routing_records
        )

        logger.debug(
            f"Retrieved: {len(asset_records)} asset, "
            f"{len(routing_records)} routing records | "
            f"tamper_alerts={tamper_alerts}"
        )

        return {
            "asset_records":   asset_records,
            "routing_records": routing_records,
            "prompt_context":  prompt_context,
            "tamper_alerts":   tamper_alerts,
        }

    def retrieve_for_condition(
        self,
        query:       str,
        condition:   str,
        k_asset:     int  = K_ASSET,
        k_routing:   int  = K_ROUTING,
        corrupt_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve records under a specific experimental condition.

        C1 — returns empty context (ungrounded LLM, no retrieval)
        C2 — retrieves from ChromaDB without hash verification;
             corrupt_ids simulates 15% tampered SQL records
        C3 — retrieves with full PureChain hash verification
        """
        if condition == "C1":
            return {
                "asset_records":   [],
                "routing_records": [],
                "prompt_context":  "",
                "tamper_alerts":   0,
            }

        verify = (condition == "C3")
        result = self.retrieve(query, k_asset, k_routing, verify=verify)

        # Simulate C2 data corruption
        if condition == "C2" and corrupt_ids:
            all_recs = result["asset_records"] + result["routing_records"]
            result["tamper_alerts"] += sum(
                1 for r in all_recs
                if r.get("chain_tx_hash", "") in corrupt_ids
            )

        return result

    # ── Parallel hash verification ─────────────────────────────

    def _tx_verified(self, tx: str) -> bool:
        """Cached verify_tx_hash: same hash is never looked up twice."""
        if tx not in self._tx_cache:
            self._tx_cache[tx] = self.pc_client.verify_tx_hash(tx)
        return self._tx_cache[tx]

    def _verify_one(self, meta: Dict[str, Any]) -> Tuple[Dict, bool]:
        """Verify a single record's hash. Returns (meta, passed)."""
        tx = meta.get("chain_tx_hash", "")
        if not tx:
            return meta, True     # empty hash → not yet on-chain, pass through
        if not self._tx_verified(tx):
            logger.warning(
                f"Tamper alert — tx={tx[:16]}... failed verification"
            )
            return meta, False
        return meta, True

    def _verify_parallel(
        self,
        asset_metas:   List[Dict],
        routing_metas: List[Dict],
    ) -> Tuple[List[Dict], List[Dict], int]:
        """
        Verify all retrieved records in parallel using a thread pool.
        Returns (verified_asset, verified_routing, tamper_alert_count).

        Non-empty hashes that hit the cache are resolved instantly without
        spawning a thread; only genuinely new hashes go to the network.
        """
        all_metas = [("asset", m) for m in asset_metas] + \
                    [("routing", m) for m in routing_metas]
        if not all_metas:
            return [], [], 0

        results: Dict[int, Tuple[str, Dict, bool]] = {}

        with ThreadPoolExecutor(max_workers=len(all_metas)) as pool:
            future_to_idx = {
                pool.submit(self._verify_one, m): (i, tag)
                for i, (tag, m) in enumerate(all_metas)
            }
            for future in as_completed(future_to_idx):
                idx, tag = future_to_idx[future]
                meta, passed = future.result()
                results[idx] = (tag, meta, passed)

        asset_records:   List[Dict] = []
        routing_records: List[Dict] = []
        tamper_alerts = 0

        for idx in sorted(results):
            tag, meta, passed = results[idx]
            if passed:
                (asset_records if tag == "asset" else routing_records).append(meta)
            else:
                tamper_alerts += 1

        return asset_records, routing_records, tamper_alerts

    # ── Utilities ──────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        return self.embed_model.encode(text).tolist()

    def _build_prompt_context(
        self,
        asset_records:   List[Dict],
        routing_records: List[Dict],
    ) -> str:
        """Format retrieved records as a structured LLM context block."""
        if not asset_records and not routing_records:
            return ""
        lines = ["=== Retrieved Supply Records (PureChain-verified) ==="]
        for i, rec in enumerate(asset_records, 1):
            lines.append(rec.get("prompt_text", f"[Asset record {i}]"))
        for i, rec in enumerate(routing_records, 1):
            lines.append(rec.get("prompt_text", f"[Routing record {i}]"))
        lines.append("=== End of Supply Records ===")
        return "\n\n".join(lines)

    @staticmethod
    def _make_id(*parts: str) -> str:
        import hashlib, time
        raw = "_".join(str(p) for p in parts) + str(time.time())
        return hashlib.md5(raw.encode()).hexdigest()

    def collection_counts(self) -> Dict[str, int]:
        return {
            "asset_records":   self.col_asset.count(),
            "routing_records": self.col_routing.count(),
        }

    def reset(self) -> None:
        """Clear all collections — use only in testing."""
        self.chroma.delete_collection(COLLECTION_ASSET)
        self.chroma.delete_collection(COLLECTION_ROUTING)
        self.col_asset = self.chroma.get_or_create_collection(
            COLLECTION_ASSET
        )
        self.col_routing = self.chroma.get_or_create_collection(
            COLLECTION_ROUTING
        )
        logger.warning("VectorIndex reset — all records cleared")


if __name__ == "__main__":
    import json
    from src.layer1.event_generator import EventGenerator

    gen    = EventGenerator(random_seed=0)
    client = get_client(mock=True)
    index  = VectorIndex(client=client, mock=True)

    rec_a = gen.generate_asset_state("B-042", "H3", 18.2, 31.0, 74.0)
    client.log_asset_state(rec_a)
    index.add_asset_record(rec_a.to_dict())

    rec_b = gen.generate_routing_condition("R1", "H1", "H3", "K2")
    client.log_routing_condition(rec_b)
    index.add_routing_record(rec_b.to_dict())

    print("Counts:", index.collection_counts())

    result = index.retrieve(
        query="Should we reorder asset B-042 at H3?",
        k_asset=1,
        k_routing=1,
    )
    print(f"\nRetrieved: {len(result['asset_records'])} asset, "
          f"{len(result['routing_records'])} routing")
    print(f"Tamper alerts: {result['tamper_alerts']}")
    print("\nPrompt context preview:")
    print(result["prompt_context"][:400])