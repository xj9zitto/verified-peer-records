"""
serialiser.py
Layer 2 — Natural language serialisation of on-chain records.

Converts Type-A (asset state) and Type-B (routing condition) records
into natural language sentences before embedding. Raw JSON is avoided
because syntax tokens dominate the embedding space and reduce semantic
retrieval quality (see Section III of the paper).

Every serialised sentence ends with the chain_tx_hash so the physics
checker can later verify that each LLM claim is traceable to a
specific on-chain transaction.
"""

from __future__ import annotations
from typing import Dict, Any, Union
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.layer1.event_generator import AssetStateRecord, RoutingConditionRecord


class RecordSerialiser:
    """
    Converts supply event records into natural language strings
    suitable for vector embedding and LLM prompt injection.

    Two serialisation targets:
      - embed_text : used for vector embedding (concise)
      - prompt_text: used for LLM context injection (verbose)
    """

    # ── Type-A serialisation ───────────────────────────────────

    def serialise_asset(
        self,
        record: Union[AssetStateRecord, Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Serialise a Type-A asset state record.

        Returns
        -------
        dict with keys:
          embed_text  : sentence for ChromaDB embedding
          prompt_text : sentence for LLM context block
          record_type : "asset_state"
        """
        r = self._normalise(record)

        tx   = r.get("chain_tx_hash", "pending")
        aid  = r.get("asset_id",       "unknown")
        hub  = r.get("hub_location",   "unknown")
        ts   = r.get("timestamp",      "unknown")
        soc  = r.get("soc_pct",        0.0)
        temp = r.get("temperature_c",  0.0)
        deg  = r.get("degradation_pct", 100.0)

        embed_text = (
            f"Asset {aid} at hub {hub} on {ts}: "
            f"SOC {soc:.1f}%, temperature {temp:.1f}C, "
            f"degradation {deg:.1f}%. "
            f"PureChain transaction: {tx}."
        )

        prompt_text = (
            f"[SUPPLY RECORD — Asset State]\n"
            f"  Asset ID   : {aid}\n"
            f"  Location   : {hub}\n"
            f"  Timestamp  : {ts}\n"
            f"  SOC        : {soc:.1f}%\n"
            f"  Temperature: {temp:.1f}°C\n"
            f"  Degradation: {deg:.1f}% remaining capacity\n"
            f"  Chain Tx   : {tx}"
        )

        return {
            "embed_text":  embed_text,
            "prompt_text": prompt_text,
            "record_type": "asset_state",
            "chain_tx_hash": tx,
            "asset_id": aid,
        }

    # ── Type-B serialisation ───────────────────────────────────

    def serialise_routing(
        self,
        record: Union[RoutingConditionRecord, Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Serialise a Type-B routing condition record.

        Returns
        -------
        dict with keys:
          embed_text  : sentence for ChromaDB embedding
          prompt_text : sentence for LLM context block
          record_type : "routing_condition"
        """
        r = self._normalise(record)

        tx        = r.get("chain_tx_hash",       "pending")
        path_id   = r.get("path_id",             "unknown")
        origin    = r.get("hop_origin",          "unknown")
        dest      = r.get("hop_dest",            "unknown")
        transit   = r.get("transit_h",           0.0)
        amb_min   = r.get("amb_temp_min_c",      0.0)
        amb_max   = r.get("amb_temp_max_c",      0.0)
        ctrl      = r.get("temp_controlled",     False)
        carrier   = r.get("carrier_id",          "unknown")
        variance  = r.get("carrier_variance_c",  0.0)

        ctrl_str = "yes" if ctrl else "no"

        embed_text = (
            f"Hop {origin} to {dest} on path {path_id}: "
            f"{transit:.0f}h transit, "
            f"ambient {amb_min:.0f}–{amb_max:.0f}C, "
            f"temperature controlled {ctrl_str}, "
            f"carrier {carrier} with +/-{variance:.0f}C variance. "
            f"PureChain transaction: {tx}."
        )

        prompt_text = (
            f"[SUPPLY RECORD — Routing Condition]\n"
            f"  Path       : {path_id}  ({origin} → {dest})\n"
            f"  Transit    : {transit:.0f} hours\n"
            f"  Ambient    : {amb_min:.0f}–{amb_max:.0f}°C\n"
            f"  Temp ctrl  : {ctrl_str}\n"
            f"  Carrier    : {carrier}  (±{variance:.0f}°C variance)\n"
            f"  Chain Tx   : {tx}"
        )

        return {
            "embed_text":  embed_text,
            "prompt_text": prompt_text,
            "record_type": "routing_condition",
            "chain_tx_hash": tx,
            "path_id": path_id,
        }

    # ── Batch serialisation ────────────────────────────────────

    def serialise_context(
        self,
        asset_records:   list,
        routing_records: list,
    ) -> str:
        """
        Serialise a mixed list of retrieved records into a single
        LLM prompt context block, ready for insertion.
        """
        lines = ["=== Retrieved Supply Records ==="]

        for rec in asset_records:
            s = self.serialise_asset(rec)
            lines.append(s["prompt_text"])

        for rec in routing_records:
            s = self.serialise_routing(rec)
            lines.append(s["prompt_text"])

        lines.append("=== End of Supply Records ===")
        return "\n\n".join(lines)

    # ── Internal helpers ───────────────────────────────────────

    @staticmethod
    def _normalise(
        record: Union[AssetStateRecord, RoutingConditionRecord, Dict]
    ) -> Dict[str, Any]:
        """Convert dataclass or dict to plain dict."""
        if isinstance(record, dict):
            return record
        if hasattr(record, "to_dict"):
            return record.to_dict()
        return record.__dict__


# Module-level singleton
serialiser = RecordSerialiser()


if __name__ == "__main__":
    from src.layer1.event_generator import EventGenerator
    gen = EventGenerator(random_seed=0)

    rec_a = gen.generate_asset_state(
        "B-042", "H3", 18.2, 31.0, 74.0
    )
    rec_b = gen.generate_routing_condition("R1", "H1", "H3", "K2")

    s = RecordSerialiser()
    sa = s.serialise_asset(rec_a)
    sb = s.serialise_routing(rec_b)

    print("Type-A embed:\n ", sa["embed_text"])
    print("\nType-B embed:\n ", sb["embed_text"])
    print("\nPrompt context:\n",
          s.serialise_context([rec_a.to_dict()], [rec_b.to_dict()]))
