"""
calce_loader.py
Dataset — NASA CALCE battery cycle data loader.

Loads and preprocesses publicly available lithium-ion battery
cycle data from the NASA CALCE dataset. Extracts SOC trajectories,
temperature readings, and degradation levels per cycle to serve as
the physical basis for synthetic scenario generation.

Dataset URL: https://calce.umd.edu/battery-data
Recommended: CS2 series (CS2_33, CS2_35, CS2_36, CS2_37, CS2_38)

Place downloaded .csv files in: data/raw/
"""

from __future__ import annotations
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import (
    RAW_DIR,
    SOC_REORDER_THRESHOLD,
    DEGRADATION_REPLACE_FLAG,
    DEGRADATION_COHORTS,
)


# ── Data classes ───────────────────────────────────────────────

@dataclass
class BatteryCycle:
    """Single charge-discharge cycle extracted from CALCE data."""
    cycle_index:     int
    soc_trajectory:  List[float]   # SOC readings through cycle (%)
    temp_trajectory: List[float]   # temperature readings (°C)
    degradation_pct: float         # remaining capacity vs nominal (%)
    min_soc:         float
    max_temp:        float


@dataclass
class BatteryDataset:
    """Processed CALCE battery dataset ready for scenario generation."""
    battery_id:        str
    nominal_capacity:  float        # Ah
    cycles:            List[BatteryCycle]
    cohort_samples:    Dict[str, List[BatteryCycle]]  # keyed by cohort name


# ── CALCE Loader ───────────────────────────────────────────────

class CALCELoader:
    """
    Loads NASA CALCE battery cycle CSV data and extracts
    SOC, temperature, and degradation trajectories.

    If real CALCE files are not available, falls back to a
    physics-based synthetic generator calibrated on published
    CALCE cycle statistics, ensuring the scenario generator
    always has data to work with.
    """

    # Published CALCE CS2 nominal capacity
    NOMINAL_CAPACITY_AH = 1.1

    def __init__(self, data_dir: Path = RAW_DIR) -> None:
        self.data_dir = Path(data_dir)
        logger.info(f"CALCELoader initialised — data dir: {self.data_dir}")

    # ── Public interface ───────────────────────────────────────

    def load(self, battery_id: str = "CS2_35") -> BatteryDataset:
        """
        Load battery dataset for the given battery ID.
        Falls back to synthetic generation if file not found.
        """
        csv_path = self.data_dir / f"{battery_id}.csv"

        if csv_path.exists():
            logger.info(f"Loading CALCE data: {csv_path}")
            cycles = self._load_from_csv(csv_path)
        else:
            logger.warning(
                f"CALCE file not found: {csv_path}. "
                "Falling back to physics-based synthetic generation. "
                "Download real data from https://calce.umd.edu/battery-data"
            )
            cycles = self._generate_synthetic_cycles()

        cohort_samples = self._split_by_cohort(cycles)

        dataset = BatteryDataset(
            battery_id=battery_id,
            nominal_capacity=self.NOMINAL_CAPACITY_AH,
            cycles=cycles,
            cohort_samples=cohort_samples,
        )
        logger.info(
            f"Dataset ready: {len(cycles)} cycles | "
            f"healthy={len(cohort_samples['healthy'])} "
            f"degraded={len(cohort_samples['degraded'])} "
            f"critical={len(cohort_samples['critical'])}"
        )
        return dataset

    def load_all(self) -> Dict[str, BatteryDataset]:
        """Load all CSV files found in data_dir."""
        csv_files = glob.glob(str(self.data_dir / "CS2_*.csv"))
        if not csv_files:
            logger.warning(
                "No CALCE CSV files found. "
                "Using synthetic generation for all batteries."
            )
            return {"synthetic": self.load("synthetic")}
        return {
            Path(f).stem: self.load(Path(f).stem)
            for f in csv_files
        }

    def sample_state(
        self,
        dataset:    BatteryDataset,
        cohort:     str,
        rng:        np.random.Generator,
    ) -> Tuple[float, float, float]:
        """
        Sample (soc, temperature, degradation) from a given cohort.

        Returns
        -------
        soc_pct         : state-of-charge %
        temperature_c   : temperature in °C
        degradation_pct : remaining capacity %
        """
        cohort_cycles = dataset.cohort_samples.get(cohort, [])
        if not cohort_cycles:
            raise ValueError(f"No cycles in cohort '{cohort}'")

        cycle = rng.choice(cohort_cycles)  # type: ignore[arg-type]

        # Sample a point along the trajectory
        idx = rng.integers(0, len(cycle.soc_trajectory))
        soc  = float(cycle.soc_trajectory[idx])
        temp = float(cycle.temp_trajectory[idx])
        deg  = cycle.degradation_pct

        return round(soc, 2), round(temp, 2), round(deg, 2)

    # ── CSV loading ────────────────────────────────────────────

    def _load_from_csv(self, path: Path) -> List[BatteryCycle]:
        """
        Parse CALCE CSV file.
        Expected columns: cycle_index, voltage, current,
                          temperature, capacity, soc
        (Exact column names vary — we attempt multiple mappings.)
        """
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()

        # Column name normalisation
        col_map = {
            "cycle": "cycle_index",
            "cycle_index": "cycle_index",
            "temp": "temperature",
            "temperature": "temperature",
            "temp_measured": "temperature",
            "soc": "soc",
            "state_of_charge": "soc",
            "capacity": "capacity",
            "discharge_capacity": "capacity",
        }
        df = df.rename(columns={
            c: col_map[c] for c in df.columns if c in col_map
        })

        required = {"cycle_index", "temperature"}
        missing  = required - set(df.columns)
        if missing:
            logger.warning(
                f"Missing columns: {missing}. "
                "Falling back to synthetic generation."
            )
            return self._generate_synthetic_cycles()

        # Compute SOC if not present
        if "soc" not in df.columns:
            if "capacity" in df.columns:
                max_cap = df["capacity"].max()
                df["soc"] = (df["capacity"] / max_cap * 100).clip(0, 100)
            else:
                df["soc"] = 100.0

        # Compute degradation from capacity fade per cycle
        cycles: List[BatteryCycle] = []
        for cycle_idx, grp in df.groupby("cycle_index"):
            cap_this = grp["capacity"].max() if "capacity" in grp else None
            if cap_this is not None:
                deg = min(100.0, (cap_this / self.NOMINAL_CAPACITY_AH) * 100)
            else:
                total = len(df["cycle_index"].unique())
                deg = max(60.0, 100.0 - float(cycle_idx) / total * 40.0)

            soc_traj  = grp["soc"].tolist()
            temp_traj = grp["temperature"].tolist()

            cycles.append(BatteryCycle(
                cycle_index=int(cycle_idx),
                soc_trajectory=soc_traj,
                temp_trajectory=temp_traj,
                degradation_pct=round(deg, 2),
                min_soc=float(min(soc_traj)),
                max_temp=float(max(temp_traj)),
            ))

        logger.info(f"Loaded {len(cycles)} cycles from {path.name}")
        return cycles

    # ── Synthetic fallback ─────────────────────────────────────

    def _generate_synthetic_cycles(
        self,
        n_cycles: int = 500,
        seed:     int = 42,
    ) -> List[BatteryCycle]:
        """
        Physics-based synthetic cycle generator calibrated on
        published NASA CALCE CS2 degradation statistics.

        Degradation model: linear fade from 100% to ~65% over
        500 cycles with Gaussian noise (std=0.5%), consistent
        with published CALCE CS2 capacity fade curves.
        """
        rng    = np.random.default_rng(seed)
        cycles = []

        for i in range(1, n_cycles + 1):
            # Capacity fade: linear + noise
            base_deg = 100.0 - (35.0 * i / n_cycles)
            deg = float(np.clip(
                base_deg + rng.normal(0, 0.5), 60.0, 100.0
            ))

            # SOC trajectory: discharge from ~95% to ~10-25%
            n_points = 50
            soc_start = float(rng.uniform(90.0, 98.0))
            soc_end   = float(rng.uniform(10.0, 25.0))
            soc_traj  = np.linspace(soc_start, soc_end, n_points)
            soc_traj += rng.normal(0, 0.3, n_points)
            soc_traj  = np.clip(soc_traj, 0.0, 100.0).tolist()

            # Temperature trajectory: rises during discharge
            temp_base  = float(rng.uniform(20.0, 30.0))
            temp_rise  = float(rng.uniform(3.0, 8.0))
            temp_traj  = np.linspace(temp_base, temp_base + temp_rise, n_points)
            temp_traj += rng.normal(0, 0.2, n_points)
            temp_traj  = temp_traj.tolist()

            cycles.append(BatteryCycle(
                cycle_index=i,
                soc_trajectory=[round(s, 2) for s in soc_traj],
                temp_trajectory=[round(t, 2) for t in temp_traj],
                degradation_pct=round(deg, 2),
                min_soc=round(float(min(soc_traj)), 2),
                max_temp=round(float(max(temp_traj)), 2),
            ))

        logger.info(
            f"Synthetic fallback: generated {len(cycles)} cycles "
            f"(deg range: "
            f"{min(c.degradation_pct for c in cycles):.1f}–"
            f"{max(c.degradation_pct for c in cycles):.1f}%)"
        )
        return cycles

    # ── Cohort splitting ───────────────────────────────────────

    def _split_by_cohort(
        self, cycles: List[BatteryCycle]
    ) -> Dict[str, List[BatteryCycle]]:
        """
        Split cycles into three degradation cohorts as defined
        in config.py: healthy (>80%), degraded (70-80%),
        critical (<70%).
        """
        result: Dict[str, List[BatteryCycle]] = {
            name: [] for name in DEGRADATION_COHORTS
        }
        for cycle in cycles:
            for cohort_name, (low, high) in DEGRADATION_COHORTS.items():
                if low <= cycle.degradation_pct < high:
                    result[cohort_name].append(cycle)
                    break
            else:
                # Catches deg == 100.0 (brand new cell)
                result["healthy"].append(cycle)

        for name, cyc_list in result.items():
            logger.debug(f"Cohort '{name}': {len(cyc_list)} cycles")

        return result


# ── Module-level instance ──────────────────────────────────────

calce_loader = CALCELoader()


if __name__ == "__main__":
    loader  = CALCELoader()
    dataset = loader.load("CS2_35")

    print(f"Battery:  {dataset.battery_id}")
    print(f"Cycles:   {len(dataset.cycles)}")
    print(f"Healthy:  {len(dataset.cohort_samples['healthy'])}")
    print(f"Degraded: {len(dataset.cohort_samples['degraded'])}")
    print(f"Critical: {len(dataset.cohort_samples['critical'])}")

    rng = np.random.default_rng(0)
    soc, temp, deg = loader.sample_state(dataset, "degraded", rng)
    print(f"\nDegraded sample: SOC={soc}%  T={temp}C  deg={deg}%")
