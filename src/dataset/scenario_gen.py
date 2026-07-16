"""
scenario_gen.py
Dataset — Synthetic scenario generator.

Generates 120 labelled evaluation scenarios (30 per type) using
physics-constrained ground truth labels derived from the battery
thresholds in config.py. Labels are computed BEFORE any LLM
involvement to prevent circular evaluation.

Fixes applied:
  v2 — Carrier naming conflict resolved (K1/K2/K3 vs C0-C3 conditions)
  v3 — Path scenario supplier-scoping (S1->R1/R2, S2->R3/R4)
  v4 — Full-hop Type-B records for reroute and path scenarios
       no_safe_route_available correctly propagated

Output: data/synthetic/scenarios.json
"""

from __future__ import annotations
import json
import uuid
import random
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import (
    SYNTHETIC_DIR, SCENARIO_COUNTS, TOTAL_SCENARIOS,
    MULTI_CONSTRAINT_N, DEGRADATION_COHORTS,
    SOC_REORDER_THRESHOLD, TEMP_ROUTING_BLOCK,
    DEGRADATION_REPLACE_FLAG, ALPHA_DEFAULT, BETA_DEFAULT,
    PI_ROUTES,
)
from src.layer1.pi_network import pi_network, PINetwork
from src.layer1.event_generator import (
    EventGenerator, AssetStateRecord, RoutingConditionRecord
)
from src.dataset.calce_loader import CALCELoader, BatteryDataset


# ── Dataclasses ────────────────────────────────────────────────

@dataclass
class Scenario:
    """
    A single evaluation scenario with ground truth label.
    The label is computed deterministically from the physics
    checker BEFORE any LLM sees the scenario.
    """
    scenario_id:         str
    scenario_type:       str
    asset_id:            str
    query:               str
    asset_records:       List[Dict[str, Any]]
    routing_records:     List[Dict[str, Any]]
    ground_truth:        str
    label:               str
    degradation_pct:     float
    soc_pct:             float
    temperature_c:       float
    route_id:            str
    carrier_id:          str
    is_multi_constraint: bool = False
    metadata:            Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Physics helpers ─────────────────────────────────────────────

def compute_t_safe(
    degradation_pct: float,
    alpha: float = ALPHA_DEFAULT,
) -> float:
    """Eq. 1: T_safe = 45 - max(0, 80 - d) * alpha"""
    return TEMP_ROUTING_BLOCK - max(
        0.0, DEGRADATION_REPLACE_FLAG - degradation_pct
    ) * alpha


def compute_t_arr(
    hop_amb_max:      float,
    carrier_variance: float,
    degradation_pct:  float,
    beta:             float = BETA_DEFAULT,
) -> float:
    """Eq. 2: T_arr = T_amb_max + V_kappa + max(0, 80 - d) * beta"""
    return hop_amb_max + carrier_variance + max(
        0.0, DEGRADATION_REPLACE_FLAG - degradation_pct
    ) * beta


def route_is_safe(
    route_id:        str,
    carrier_id:      str,
    degradation_pct: float,
    network:         PINetwork = pi_network,
    alpha:           float = ALPHA_DEFAULT,
    beta:            float = BETA_DEFAULT,
) -> Tuple[bool, float]:
    """Return (is_safe, max_projected_temp) for a route-carrier pair."""
    t_safe   = compute_t_safe(degradation_pct, alpha)
    carrier  = network.get_carrier(carrier_id)
    max_temp = 0.0
    for hop in network.get_route_hops(route_id):
        t_arr    = compute_t_arr(hop.amb_max, carrier.variance, degradation_pct, beta)
        max_temp = max(max_temp, t_arr)
    return max_temp <= t_safe, max_temp


# ── Scenario Generator ─────────────────────────────────────────

class ScenarioGenerator:
    """
    Generates 120 labelled evaluation scenarios across four types.
    Uses CALCE battery data for realistic state sampling and applies
    physics-based constraints for deterministic ground truth labelling.
    """

    def __init__(
        self,
        seed:    int = 42,
        network: PINetwork = pi_network,
    ) -> None:
        self.rng     = np.random.default_rng(seed)
        self.py_rng  = random.Random(seed)
        self.network = network
        self.loader  = CALCELoader()
        self.evgen   = EventGenerator(network=network, random_seed=seed)
        self.dataset: Optional[BatteryDataset] = None

    def _load_dataset(self) -> None:
        if self.dataset is None:
            self.dataset = self.loader.load()

    def generate_all(self, save: bool = True) -> List[Scenario]:
        """Generate all 120 scenarios and optionally save to disk."""
        self._load_dataset()
        scenarios: List[Scenario] = []

        scenarios += self._generate_reorder(SCENARIO_COUNTS["reorder"])
        scenarios += self._generate_reroute(SCENARIO_COUNTS["reroute"])
        scenarios += self._generate_path(SCENARIO_COUNTS["path"])
        scenarios += self._generate_carrier(SCENARIO_COUNTS["carrier"])

        mc_indices = self.py_rng.sample(range(len(scenarios)), MULTI_CONSTRAINT_N)
        for i in mc_indices:
            scenarios[i].is_multi_constraint = True

        logger.info(
            f"Generated {len(scenarios)} scenarios "
            f"({MULTI_CONSTRAINT_N} multi-constraint tagged)"
        )
        if save:
            self._save(scenarios)
        return scenarios

    # ── Internal helper: full route trajectory ─────────────────

    def _route_full_trajectory(
        self,
        route_id:    str,
        prefer_ctrl: bool = False,
        force_unctr: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate Type-B records for ALL hops of a route.

        prefer_ctrl=True  -> use K3 (controlled) wherever available.
                             Used for the safe/good route.
        force_unctr=True  -> use uncontrolled carrier wherever possible.
                             Used for the unsafe/bad route.
        Falls back to any available carrier when preference is not met.
        """
        records = []
        for hop_o, hop_d in self.network.get_route(route_id).hops:
            avail = self.network.get_hop(hop_o, hop_d).carriers
            unctr = [c for c in avail
                     if not self.network.get_carrier(c).controlled]
            ctrl  = [c for c in avail
                     if self.network.get_carrier(c).controlled]

            if force_unctr and unctr:
                cid = self.py_rng.choice(unctr)
            elif prefer_ctrl and ctrl:
                cid = ctrl[0]
            else:
                cid = avail[-1]

            records.append(
                self.evgen.generate_routing_condition(
                    path_id=route_id,
                    hop_origin=hop_o,
                    hop_dest=hop_d,
                    carrier_id=cid,
                ).to_dict()
            )
        return records

    # ── Reorder scenarios ──────────────────────────────────────

    def _generate_reorder(self, n: int) -> List[Scenario]:
        """
        30 scenarios testing SOC-threshold reorder decisions.
        Half below threshold (reorder), half above (hold or
        reorder_degradation_flag). A single final-hop Type-B record
        is sufficient — reorder decisions do not depend on routing.
        """
        scenarios = []
        cohorts   = ["healthy", "degraded", "critical"]

        for i in range(n):
            cohort     = cohorts[i % len(cohorts)]
            asset_id   = f"B-{100 + i:03d}"
            route_id   = self.py_rng.choice(["R1", "R2", "R3", "R4"])
            carrier_id = self.py_rng.choice(
                self.network.get_route_hops(route_id)[-1].carriers
            )

            soc, temp, deg = self.loader.sample_state(
                self.dataset, cohort, self.rng
            )

            if i < n // 2:
                soc            = round(self.rng.uniform(5.0, SOC_REORDER_THRESHOLD - 0.5), 2)
                correct_action = "reorder"
            else:
                soc            = round(self.rng.uniform(SOC_REORDER_THRESHOLD + 1.0, 85.0), 2)
                correct_action = "hold" if deg >= DEGRADATION_REPLACE_FLAG \
                    else "reorder_degradation_flag"

            hub_location = self.network.get_route(route_id).nodes[-2]

            asset_rec   = self.evgen.generate_asset_state(
                asset_id=asset_id,
                hub_location=hub_location,
                soc_pct=soc,
                temperature_c=temp,
                degradation_pct=deg,
            )
            routing_rec = self.evgen.generate_routing_condition(
                path_id=route_id,
                hop_origin=hub_location,
                hop_dest=self.network.get_route(route_id).nodes[-1],
                carrier_id=carrier_id,
            )

            query = (
                f"Battery asset {asset_id} is currently at "
                f"{hub_location}. Based on the supply records, "
                f"should we initiate a reorder, or hold current inventory?"
            )

            scenarios.append(Scenario(
                scenario_id=str(uuid.uuid4()),
                scenario_type="reorder",
                asset_id=asset_id,
                query=query,
                asset_records=[asset_rec.to_dict()],
                routing_records=[routing_rec.to_dict()],
                ground_truth=correct_action,
                label="valid_action",
                degradation_pct=deg,
                soc_pct=soc,
                temperature_c=temp,
                route_id=route_id,
                carrier_id=carrier_id,
            ))

        logger.info(f"Reorder scenarios: {len(scenarios)}")
        return scenarios

    # ── Reroute scenarios ──────────────────────────────────────

    def _generate_reroute(self, n: int) -> List[Scenario]:
        """
        30 scenarios testing mid-transit temperature violation rerouting.

        Full Type-B records are generated for ALL hops of BOTH routes
        so the LLM can compute the complete temperature trajectory for
        each route and compare them accurately.

          Bad route  -> uncontrolled carriers (K1/K2 where available)
          Good route -> controlled carriers preferred (K3 where available)
        """
        scenarios   = []
        route_pairs = [("R1", "R2"), ("R3", "R4")]

        for i in range(n):
            cohort   = "degraded" if i < n // 2 else "critical"
            asset_id = f"B-{200 + i:03d}"

            soc, temp, deg = self.loader.sample_state(
                self.dataset, cohort, self.rng
            )

            bad_route_id, good_route_id = self.py_rng.choice(route_pairs)

            bad_safe,  bad_max  = route_is_safe(bad_route_id,  "K1", deg)
            good_safe, good_max = route_is_safe(good_route_id, "K3", deg)
            t_safe              = compute_t_safe(deg)

            hub_location = self.network.get_route(bad_route_id).nodes[0]

            asset_rec = self.evgen.generate_asset_state(
                asset_id=asset_id,
                hub_location=hub_location,
                soc_pct=soc,
                temperature_c=temp,
                degradation_pct=deg,
            )

            # Full trajectory for bad route (uncontrolled)
            bad_records  = self._route_full_trajectory(bad_route_id,  force_unctr=True)
            # Full trajectory for good route (controlled preferred)
            good_records = self._route_full_trajectory(good_route_id, prefer_ctrl=True)

            routing_records = bad_records + good_records
            # Determine correct action:
            # If bad is unsafe AND good is safe → take good route
            # If bad is unsafe AND good is also unsafe → no safe route exists
            # If bad is safe → take bad route (shouldn't happen by design, but safe fallback)
            if not bad_safe and good_safe:
                correct_action = good_route_id
            elif not bad_safe and not good_safe:
                correct_action = "no_safe_route_available"
            else:
                correct_action = bad_route_id

            query = (
                f"Asset {asset_id} is currently at {hub_location} "
                f"and needs to reach the factory. "
                f"Given the full routing conditions for {bad_route_id} "
                f"and {good_route_id}, which route should be taken?"
            )

            scenarios.append(Scenario(
                scenario_id=str(uuid.uuid4()),
                scenario_type="reroute",
                asset_id=asset_id,
                query=query,
                asset_records=[asset_rec.to_dict()],
                routing_records=routing_records,
                ground_truth=correct_action,
                label="valid_action",
                degradation_pct=deg,
                soc_pct=soc,
                temperature_c=temp,
                route_id=bad_route_id,
                carrier_id="K1",
                metadata={
                    "bad_route_safe":  bad_safe,
                    "bad_max_temp":    round(bad_max, 2),
                    "good_max_temp":   round(good_max, 2),
                    "t_safe":          round(t_safe, 2),
                    "bad_route_hops":  len(bad_records),
                    "good_route_hops": len(good_records),
                },
            ))

        logger.info(f"Reroute scenarios: {len(scenarios)}")
        return scenarios

    # ── Path selection scenarios ───────────────────────────────

    def _generate_path(self, n: int) -> List[Scenario]:
        """
        30 scenarios testing multi-hop path selection.

        Routes evaluated are scoped to the asset's supplier:
          S1 assets -> R1, R2 only
          S2 assets -> R3, R4 only

        Full Type-B records are generated for ALL hops of each
        accessible route so the LLM can see the complete temperature
        profile per route and make an informed comparison.

        When no route is thermally safe (heavily degraded batteries),
        the correct action is 'no_safe_route_available' — the LLM
        should recommend escalation rather than choosing a route.
        """
        scenarios = []

        supplier_routes = {
            "S1": [rid for rid, nodes in PI_ROUTES if nodes[0] == "S1"],
            "S2": [rid for rid, nodes in PI_ROUTES if nodes[0] == "S2"],
        }

        for i in range(n):
            asset_id     = f"B-{300 + i:03d}"
            hub_location = "S1" if i % 2 == 0 else "S2"
            cohort       = self.py_rng.choice(
                ["healthy", "degraded", "degraded", "critical"]
            )

            soc, temp, deg = self.loader.sample_state(
                self.dataset, cohort, self.rng
            )

            accessible = supplier_routes[hub_location]

            route_safety = {}
            for route_id in accessible:
                safe, max_t = route_is_safe(route_id, "K3", deg)
                route_safety[route_id] = {
                    "safe":       safe,
                    "max_temp":   round(max_t, 2),
                    "total_time": self.network.get_route_total_time(route_id),
                    "t_safe":     round(compute_t_safe(deg), 2),
                }

            safe_routes = [r for r, v in route_safety.items() if v["safe"]]
            correct_action = (
                min(safe_routes, key=lambda r: route_safety[r]["total_time"])
                if safe_routes else "no_safe_route_available"
            )

            asset_rec = self.evgen.generate_asset_state(
                asset_id=asset_id,
                hub_location=hub_location,
                soc_pct=soc,
                temperature_c=temp,
                degradation_pct=deg,
            )

            # Full trajectory for every hop of every accessible route
            routing_records = []
            for route_id in accessible:
                routing_records += self._route_full_trajectory(
                    route_id, prefer_ctrl=True
                )

            route_list = " or ".join(accessible)
            query = (
                f"Asset {asset_id} is ready to ship from {hub_location}. "
                f"Which route ({route_list}) should be selected given the "
                f"current battery state and the full routing conditions? "
                f"If no route is thermally safe, state that explicitly."
            )

            scenarios.append(Scenario(
                scenario_id=str(uuid.uuid4()),
                scenario_type="path",
                asset_id=asset_id,
                query=query,
                asset_records=[asset_rec.to_dict()],
                routing_records=routing_records,
                ground_truth=correct_action,
                label="valid_action",
                degradation_pct=deg,
                soc_pct=soc,
                temperature_c=temp,
                route_id=correct_action,
                carrier_id="K3",
                metadata={
                    "hub_location":      hub_location,
                    "accessible_routes": accessible,
                    "route_safety":      route_safety,
                },
            ))

        logger.info(f"Path selection scenarios: {len(scenarios)}")
        return scenarios

    # ── Carrier assignment scenarios ───────────────────────────

    def _generate_carrier(self, n: int) -> List[Scenario]:
        """
        30 scenarios testing carrier assignment at a single hop.
        Correct carrier is the safest available (lowest T_arr vs T_safe).
        """
        scenarios = []

        for i in range(n):
            asset_id = f"B-{400 + i:03d}"
            cohort   = "degraded" if i % 2 == 0 else "critical"

            soc, temp, deg = self.loader.sample_state(
                self.dataset, cohort, self.rng
            )

            route_id = self.py_rng.choice(["R1", "R2", "R3", "R4"])
            route    = self.network.get_route(route_id)
            hop_pair = route.hops[0]
            hop      = self.network.get_hop(hop_pair[0], hop_pair[1])
            avail    = hop.carriers

            carrier_results = {}
            for cid in avail:
                carrier = self.network.get_carrier(cid)
                t_arr   = compute_t_arr(hop.amb_max, carrier.variance, deg)
                t_safe  = compute_t_safe(deg)
                carrier_results[cid] = {
                    "t_arr":  round(t_arr, 2),
                    "t_safe": round(t_safe, 2),
                    "safe":   t_arr <= t_safe,
                }

            safe_carriers  = [c for c, v in carrier_results.items() if v["safe"]]
            correct_action = safe_carriers[0] if safe_carriers else "K3"

            asset_rec = self.evgen.generate_asset_state(
                asset_id=asset_id,
                hub_location=hop_pair[0],
                soc_pct=soc,
                temperature_c=temp,
                degradation_pct=deg,
            )

            routing_records = [
                self.evgen.generate_routing_condition(
                    path_id=route_id,
                    hop_origin=hop_pair[0],
                    hop_dest=hop_pair[1],
                    carrier_id=cid,
                ).to_dict()
                for cid in avail
            ]

            query = (
                f"Asset {asset_id} is at {hop_pair[0]} ready for the next "
                f"hop to {hop_pair[1]}. Available carriers are {avail}. "
                f"Which carrier should be assigned?"
            )

            scenarios.append(Scenario(
                scenario_id=str(uuid.uuid4()),
                scenario_type="carrier",
                asset_id=asset_id,
                query=query,
                asset_records=[asset_rec.to_dict()],
                routing_records=routing_records,
                ground_truth=correct_action,
                label="valid_action",
                degradation_pct=deg,
                soc_pct=soc,
                temperature_c=temp,
                route_id=route_id,
                carrier_id=correct_action,
                metadata={"carrier_results": carrier_results},
            ))

        logger.info(f"Carrier assignment scenarios: {len(scenarios)}")
        return scenarios

    # ── Save / Load ────────────────────────────────────────────

    def _save(self, scenarios: List[Scenario]) -> None:
        out_path = Path(SYNTHETIC_DIR) / "scenarios.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([s.to_dict() for s in scenarios], f, indent=2)
        logger.success(f"Scenarios saved: {out_path} ({len(scenarios)} total)")

    @staticmethod
    def load_scenarios(path: Optional[Path] = None) -> List[Dict]:
        if path is None:
            path = Path(SYNTHETIC_DIR) / "scenarios.json"
        with open(path) as f:
            return json.load(f)


if __name__ == "__main__":
    gen       = ScenarioGenerator(seed=42)
    scenarios = gen.generate_all(save=True)

    types: Dict[str, int] = {}
    for s in scenarios:
        types[s.scenario_type] = types.get(s.scenario_type, 0) + 1

    print(f"\nTotal: {len(scenarios)} scenarios")
    for t, c in types.items():
        print(f"  {t}: {c}")
    mc = sum(1 for s in scenarios if s.is_multi_constraint)
    print(f"  multi-constraint tagged: {mc}")

    # Verify routing record counts — confirm full trajectories are present
    from collections import defaultdict
    counts: Dict[str, list] = defaultdict(list)
    for s in scenarios:
        counts[s.scenario_type].append(len(s.routing_records))
    print("\nRouting records per scenario (avg / min / max):")
    for t, cs in counts.items():
        print(f"  {t}: {sum(cs)/len(cs):.1f} avg  ({min(cs)}–{max(cs)})")