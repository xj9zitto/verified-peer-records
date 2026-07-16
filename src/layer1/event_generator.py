"""
event_generator.py
Layer 1 — IIoT supply event generation.

Generates Type-A (asset state) and Type-B (routing condition) records
that are pushed to the PureChain smart contract in Layer 2.
Also used by the synthetic dataset generator to produce labelled
evaluation scenarios.
"""

from __future__ import annotations
import random
import time
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import (
    CARRIERS, SOC_REORDER_THRESHOLD,
    TEMP_ROUTING_BLOCK, DEGRADATION_REPLACE_FLAG,
    ALPHA_DEFAULT, BETA_DEFAULT,
)
from src.layer1.pi_network import PINetwork, Hop, Carrier, pi_network


# ── Record dataclasses ─────────────────────────────────────────

@dataclass
class AssetStateRecord:
    """
    Type-A record: asset state at a PI hop.
    Corresponds to on-chain log entry from SupplyEventLogger.
    """
    record_type:   str = "asset_state"
    asset_id:      str = ""
    timestamp:     str = ""
    hub_location:  str = ""
    soc_pct:       float = 0.0     # state-of-charge %
    temperature_c: float = 0.0     # temperature in °C
    degradation_pct: float = 100.0 # remaining capacity %
    sensor_hash:   str = ""
    chain_tx_hash: str = ""        # filled after PureChain logging

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_nl(self) -> str:
        """Natural language serialisation for vector embedding."""
        return (
            f"Asset {self.asset_id} at hub {self.hub_location} "
            f"on {self.timestamp}: "
            f"SOC {self.soc_pct:.1f}%, "
            f"temperature {self.temperature_c:.1f}C, "
            f"degradation {self.degradation_pct:.1f}%. "
            f"PureChain transaction: {self.chain_tx_hash}."
        )


@dataclass
class RoutingConditionRecord:
    """
    Type-B record: routing condition at a PI hop.
    Corresponds to on-chain log entry from SupplyEventLogger.
    """
    record_type:       str = "routing_condition"
    path_id:           str = ""
    hop_origin:        str = ""
    hop_dest:          str = ""
    transit_h:         float = 0.0
    amb_temp_min_c:    float = 0.0
    amb_temp_max_c:    float = 0.0
    temp_controlled:   bool = False
    carrier_id:        str = ""
    carrier_variance_c: float = 0.0
    chain_tx_hash:     str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_nl(self) -> str:
        """Natural language serialisation for vector embedding."""
        ctrl = "yes" if self.temp_controlled else "no"
        return (
            f"Hop {self.hop_origin} to {self.hop_dest} "
            f"on path {self.path_id}: "
            f"{self.transit_h:.0f}h transit, "
            f"ambient {self.amb_temp_min_c:.0f}–"
            f"{self.amb_temp_max_c:.0f}C, "
            f"temperature controlled {ctrl}, "
            f"carrier {self.carrier_id} "
            f"with +/-{self.carrier_variance_c:.0f}C variance. "
            f"PureChain transaction: {self.chain_tx_hash}."
        )


# ── Event generator ────────────────────────────────────────────

class EventGenerator:
    """
    Generates Type-A and Type-B supply event records for the PI network.

    Used in two modes:
      1. Live mode — generates events from simulated battery state
         and pushes to PureChain via the Layer 2 client.
      2. Synthetic mode — generates labelled scenarios for evaluation,
         driven by the scenario_gen.py dataset generator.
    """

    def __init__(
        self,
        network: PINetwork = pi_network,
        random_seed: Optional[int] = None,
    ) -> None:
        self.network = network
        self.rng = random.Random(random_seed)
        logger.info("EventGenerator initialised")

    # ── Type-A generation ──────────────────────────────────────

    def generate_asset_state(
        self,
        asset_id:        str,
        hub_location:    str,
        soc_pct:         float,
        temperature_c:   float,
        degradation_pct: float,
        timestamp:       Optional[str] = None,
    ) -> AssetStateRecord:
        """
        Generate a Type-A asset state record.

        Parameters
        ----------
        asset_id        : unique battery batch identifier
        hub_location    : current PI node (e.g. "H3")
        soc_pct         : state-of-charge percentage
        temperature_c   : current temperature in °C
        degradation_pct : remaining capacity percentage
        timestamp       : ISO timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        sensor_data = (
            f"{asset_id}{hub_location}"
            f"{soc_pct:.2f}{temperature_c:.2f}"
            f"{degradation_pct:.2f}{timestamp}"
        )
        sensor_hash = hashlib.sha256(
            sensor_data.encode()
        ).hexdigest()

        record = AssetStateRecord(
            asset_id=asset_id,
            timestamp=timestamp,
            hub_location=hub_location,
            soc_pct=round(soc_pct, 2),
            temperature_c=round(temperature_c, 2),
            degradation_pct=round(degradation_pct, 2),
            sensor_hash=sensor_hash,
        )
        logger.debug(
            f"Type-A generated: {asset_id} @ {hub_location} "
            f"SOC={soc_pct:.1f}% T={temperature_c:.1f}C "
            f"deg={degradation_pct:.1f}%"
        )
        return record

    # ── Type-B generation ──────────────────────────────────────

    def generate_routing_condition(
        self,
        path_id:    str,
        hop_origin: str,
        hop_dest:   str,
        carrier_id: str,
    ) -> RoutingConditionRecord:
        """
        Generate a Type-B routing condition record for a specific hop
        and carrier assignment.

        Parameters
        ----------
        path_id    : route identifier (R1–R4)
        hop_origin : origin node of the hop
        hop_dest   : destination node of the hop
        carrier_id : assigned carrier (K1/K2/K3)
        """
        hop     = self.network.get_hop(hop_origin, hop_dest)
        carrier = self.network.get_carrier(carrier_id)

        if carrier_id not in hop.carriers:
            raise ValueError(
                f"Carrier {carrier_id} not available at hop "
                f"{hop_origin}->{hop_dest}. "
                f"Available: {hop.carriers}"
            )

        record = RoutingConditionRecord(
            path_id=path_id,
            hop_origin=hop_origin,
            hop_dest=hop_dest,
            transit_h=hop.transit_h,
            amb_temp_min_c=hop.amb_min,
            amb_temp_max_c=hop.amb_max,
            temp_controlled=hop.temp_ctrl,
            carrier_id=carrier_id,
            carrier_variance_c=carrier.variance,
        )
        logger.debug(
            f"Type-B generated: {path_id} "
            f"{hop_origin}->{hop_dest} "
            f"carrier={carrier_id} "
            f"variance=+/-{carrier.variance}C"
        )
        return record

    # ── Batch generation helpers ───────────────────────────────

    def generate_route_records(
        self,
        asset_id:        str,
        route_id:        str,
        carrier_id:      str,
        initial_soc:     float,
        initial_temp:    float,
        degradation_pct: float,
        alpha:           float = ALPHA_DEFAULT,
        beta:            float = BETA_DEFAULT,
    ) -> List[AssetStateRecord | RoutingConditionRecord]:
        """
        Generate the full set of Type-A and Type-B records
        for an asset traversing a complete route.

        Returns alternating Type-B (routing) and Type-A (state)
        records for each hop, simulating progressive battery
        state changes along the route.
        """
        records: List[AssetStateRecord | RoutingConditionRecord] = []
        route   = self.network.get_route(route_id)
        soc     = initial_soc
        temp    = initial_temp
        deg     = degradation_pct

        for origin, dest in route.hops:
            hop     = self.network.get_hop(origin, dest)
            carrier = self.network.get_carrier(carrier_id)

            # Type-B: routing condition at this hop
            routing_rec = self.generate_routing_condition(
                path_id=route_id,
                hop_origin=origin,
                hop_dest=dest,
                carrier_id=carrier_id,
            )
            records.append(routing_rec)

            # Simulate battery state change during transit
            # SOC decreases ~2-5% per hour
            soc_drain = self.rng.uniform(2.0, 5.0) * hop.transit_h
            soc = max(0.0, soc - soc_drain)

            # Temperature: ambient + carrier variance (Eq. 2)
            deg_factor = max(0.0, DEGRADATION_REPLACE_FLAG - deg) * beta
            sampled_amb = self.rng.uniform(hop.amb_min, hop.amb_max)
            temp = sampled_amb + carrier.variance + deg_factor

            # Type-A: asset state on arrival at dest node
            state_rec = self.generate_asset_state(
                asset_id=asset_id,
                hub_location=dest,
                soc_pct=soc,
                temperature_c=temp,
                degradation_pct=deg,
            )
            records.append(state_rec)

        logger.info(
            f"Route records generated: {asset_id} via {route_id} "
            f"({len(records)} records total)"
        )
        return records

    def sample_temperature_at_hop(
        self,
        hop:             Hop,
        carrier:         Carrier,
        degradation_pct: float,
        beta:            float = BETA_DEFAULT,
    ) -> float:
        """
        Sample a realistic arrival temperature for a hop.
        Uses Eq. 2: T_arr = T_amb_max + V_kappa + max(0,80-d)*beta
        """
        deg_factor = max(0.0, DEGRADATION_REPLACE_FLAG - degradation_pct) * beta
        return hop.amb_max + carrier.variance + deg_factor


# ── Module-level singleton ─────────────────────────────────────

event_generator = EventGenerator()


if __name__ == "__main__":
    # Quick smoke test
    gen = EventGenerator(random_seed=42)

    # Generate a Type-A record
    rec_a = gen.generate_asset_state(
        asset_id="B-042",
        hub_location="H3",
        soc_pct=18.2,
        temperature_c=31.0,
        degradation_pct=74.0,
    )
    print("Type-A NL:", rec_a.to_nl())

    # Generate a Type-B record
    rec_b = gen.generate_routing_condition(
        path_id="R1",
        hop_origin="H1",
        hop_dest="H3",
        carrier_id="K2",
    )
    print("Type-B NL:", rec_b.to_nl())

    # Generate full route records
    records = gen.generate_route_records(
        asset_id="B-042",
        route_id="R1",
        carrier_id="K2",
        initial_soc=65.0,
        initial_temp=30.0,
        degradation_pct=74.0,
    )
    print(f"\nFull route: {len(records)} records generated")
    for r in records:
        print(f"  [{r.record_type}]", r.to_nl()[:80])
