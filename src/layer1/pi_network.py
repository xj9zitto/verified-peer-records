"""
pi_network.py
Layer 1 — Physical Internet network topology.

Defines the two-echelon PI network: 2 suppliers, 4 hubs, 1 factory,
4 end-to-end routes, and 3 carrier types (kappa_1/2/3).
All topology values come from config.py — do not hardcode here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
from loguru import logger
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import PI_HOPS, PI_ROUTES, CARRIERS


# ── Data classes ───────────────────────────────────────────────

@dataclass
class Hop:
    """Single hop between two PI nodes."""
    origin:    str
    dest:      str
    transit_h: float
    amb_min:   float
    amb_max:   float
    temp_ctrl: bool
    carriers:  List[str]

    @property
    def key(self) -> str:
        return f"{self.origin}->{self.dest}"


@dataclass
class Carrier:
    """PI carrier with temperature variance properties."""
    carrier_id: str
    ctype:      str
    variance:   float
    controlled: bool


@dataclass
class Route:
    """End-to-end PI route composed of sequential hops."""
    route_id: str
    nodes:    List[str]

    @property
    def hops(self) -> List[Tuple[str, str]]:
        return [(self.nodes[i], self.nodes[i + 1])
                for i in range(len(self.nodes) - 1)]

    @property
    def supplier(self) -> str:
        return self.nodes[0]


# ── PI Network ─────────────────────────────────────────────────

class PINetwork:
    """
    Physical Internet network for smart factory supply management.
    Instantiates the two-echelon topology from config.py.
    """

    def __init__(self) -> None:
        self._hops:     Dict[str, Hop]     = {}
        self._routes:   Dict[str, Route]   = {}
        self._carriers: Dict[str, Carrier] = {}
        self._build()
        logger.info(
            f"PINetwork ready — {len(self._hops)} hops, "
            f"{len(self._routes)} routes, "
            f"{len(self._carriers)} carrier types"
        )

    def _build(self) -> None:
        for cid, props in CARRIERS.items():
            self._carriers[cid] = Carrier(
                carrier_id=cid,
                ctype=props["type"],
                variance=props["variance"],
                controlled=props["controlled"],
            )

        for row in PI_HOPS:
            origin, dest, transit_h, amb_min, amb_max, ctrl, carriers = row
            hop = Hop(
                origin=origin, dest=dest,
                transit_h=transit_h,
                amb_min=float(amb_min), amb_max=float(amb_max),
                temp_ctrl=ctrl, carriers=carriers,
            )
            self._hops[hop.key] = hop

        for route_id, nodes in PI_ROUTES:
            self._routes[route_id] = Route(route_id=route_id, nodes=nodes)

    # ── Queries ────────────────────────────────────────────────

    def get_hop(self, origin: str, dest: str) -> Hop:
        key = f"{origin}->{dest}"
        if key not in self._hops:
            raise KeyError(f"No hop defined: {key}")
        return self._hops[key]

    def get_route(self, route_id: str) -> Route:
        if route_id not in self._routes:
            raise KeyError(f"Unknown route: {route_id}")
        return self._routes[route_id]

    def get_carrier(self, carrier_id: str) -> Carrier:
        if carrier_id not in self._carriers:
            raise KeyError(f"Unknown carrier: {carrier_id}")
        return self._carriers[carrier_id]

    def get_route_hops(self, route_id: str) -> List[Hop]:
        route = self.get_route(route_id)
        return [self.get_hop(o, d) for o, d in route.hops]

    def get_route_total_time(self, route_id: str) -> float:
        return sum(h.transit_h for h in self.get_route_hops(route_id))

    def get_route_max_ambient(self, route_id: str) -> float:
        return max(h.amb_max for h in self.get_route_hops(route_id))

    def routes_from_supplier(self, supplier: str) -> List[Route]:
        return [r for r in self._routes.values()
                if r.supplier == supplier]

    def available_carriers_at_hop(
        self, origin: str, dest: str
    ) -> List[Carrier]:
        hop = self.get_hop(origin, dest)
        return [self._carriers[cid] for cid in hop.carriers]

    @property
    def all_routes(self) -> List[Route]:
        return list(self._routes.values())

    @property
    def all_hops(self) -> List[Hop]:
        return list(self._hops.values())

    def summary(self) -> str:
        lines = ["PI Network Summary", "=" * 40]
        lines.append(f"Hops ({len(self._hops)}):")
        for hop in self.all_hops:
            ctrl = "CTRL" if hop.temp_ctrl else "UNCTR"
            lines.append(
                f"  {hop.key:15s}  {hop.transit_h}h  "
                f"{hop.amb_min}–{hop.amb_max}°C  "
                f"{ctrl}  carriers={hop.carriers}"
            )
        lines.append(f"\nRoutes ({len(self._routes)}):")
        for route in self.all_routes:
            t = self.get_route_total_time(route.route_id)
            lines.append(
                f"  {route.route_id}: "
                f"{' -> '.join(route.nodes)}  ({t}h)"
            )
        lines.append(f"\nCarriers ({len(self._carriers)}):")
        for c in self.all_carriers:
            lines.append(
                f"  {c.carrier_id}: {c.ctype}  "
                f"variance=+/-{c.variance}C  "
                f"controlled={c.controlled}"
            )
        return "\n".join(lines)


# Module-level singleton — import this everywhere
pi_network = PINetwork()


if __name__ == "__main__":
    print(pi_network.summary())
