"""
v14_loader.py — adapter from synth-batt-data v14.0 parquet to scenario dicts.

Reads:
  D:/VibeCodingProjects/synth-batt-data/data/{scenario}_{obs,labels}.parquet
  D:/VibeCodingProjects/synth-batt-data/data/run_metadata.parquet
  D:/VibeCodingProjects/synth-batt-data/data/canonical_split.json

Produces a list of "scenario instance" dicts, one per (run_id, t_decision)
sample, in a format compatible with the existing experimental harness.

Each scenario instance has:
  scenario_id     : "v14-<run_id>-<t_decision>"
  scenario_type   : "v14_safety"
  asset_id        : "B-<short>"
  query           : natural-language query for the agent
  ground_truth    : action label (hold / monitor / reorder_replacement /
                    isolate_immediately / escalate_intervention)
  metadata        : v14_run_id, scenario, cause_family, failure_stage,
                    prevention_label_*, t_decision_s, run_duration_s
  obs_window      : 60s observable summary (mean/max/range/trend per channel)
  asset_records   : list of verified peer-trajectory records with outcomes
  routing_records : empty (kept for schema compatibility)

Usage:
    python -m src.dataset.v14_loader \\
        --n 240 --out data/synthetic/v14_scenarios.json
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

V14_ROOT = Path(r"D:\VibeCodingProjects\synth-batt-data\data")

CAUSE_FAMILIES = {
    0: ["normal_cycle", "aged_normal_cycle"],
    1: ["overcharge", "moderate_overcharge", "fast_charge_abuse"],
    2: ["overdischarge"],
    3: ["mild_thermal_stress", "thermal_abuse"],
    4: ["isc", "partial_isc_stress", "isc_physical", "external_short_circuit"],
    5: ["mechanical_abuse"],
    6: ["combined", "overdischarge_fast_recharge"],
    7: ["lithium_plating"],
}

FAMILY_NAMES = {
    0: "normal_cycle",
    1: "electrical_overcharge",
    2: "electrical_overdischarge",
    3: "thermal_stressor",
    4: "short_circuit",
    5: "mechanical_abuse",
    6: "combined_stressor",
    7: "lithium_plating",
}

# Observable channels included in the LLM-visible summary
OBS_CHANNELS_PRIMARY = [
    "voltage_V_measured",
    "current_A_measured",
    "temperature_C_measured",
    "SOC_with_drift",
    "dT_dt_meas_C_per_s",
    "dV_dt_meas_V_per_s",
]

WINDOW_SECONDS = 60.0   # length of observable window before t_decision
T_FUTURE_300   = 300.0  # short prevention horizon
T_BUFFER_END   = 300.0  # require t_decision <= run_duration - T_BUFFER_END
T_MIN          = 60.0   # don't sample t earlier than this (need warm-up)

N_PEERS        = 5      # number of verified peer trajectories per scenario

ACTION_MENU = [
    "safe",      # continue or monitor — low immediate risk
    "replace",   # schedule replacement battery — warning trending toward escalation
    "urgent",    # take offline now — critical or failure in progress
]


# ── Ground-truth action mapping (3-class) ──────────────────────

def _gt_action(stage: int, p300: int, p3600: int) -> str:
    """
    Deterministic mapping (failure_stage, prevention_300, prevention_3600)
    -> action.  Three-class collapse:
        urgent  — Stage>=2 (critical or failure in progress)
        replace — Stage=1 with imminent Stage>=2 in next 5 min
        safe    — everything else (Stage=0 OR Stage=1 with no near-term escalation)
    """
    if stage >= 2:
        return "urgent"
    if stage == 1 and p300 == 1:
        return "replace"
    return "safe"


# ── Window summarisation ───────────────────────────────────────

def _summarise_window(window: pd.DataFrame) -> Dict[str, Any]:
    """Per-channel summary statistics over the observable window."""
    summary = {}
    for ch in OBS_CHANNELS_PRIMARY:
        if ch not in window.columns:
            continue
        s = window[ch].dropna()
        if len(s) == 0:
            continue
        first, last = s.iloc[0], s.iloc[-1]
        trend = "rising" if last > first + 1e-6 else \
                "falling" if last < first - 1e-6 else "flat"
        summary[ch] = {
            "mean": round(float(s.mean()), 4),
            "min":  round(float(s.min()),  4),
            "max":  round(float(s.max()),  4),
            "first":round(float(first),    4),
            "last": round(float(last),     4),
            "trend": trend,
        }
    return summary


# ── Peer corpus + retrieval ────────────────────────────────────

def _build_peer_corpus(
    scenarios:    List[str],
    train_runs:   Dict[str, set],
    md:           pd.DataFrame,
    obs_root:     Path,
    samples_per_scenario: int = 30,
    rng:          random.Random = None,
) -> List[Dict[str, Any]]:
    """
    Produce a corpus of historical peer summaries: per scenario family, sample
    `samples_per_scenario` runs from the train split, then for each run pick a
    representative observable snapshot at ~30% through the run, plus the run's
    final outcome.  Each peer carries a chain_tx_hash for PureChain logging.
    """
    rng = rng or random.Random(42)
    peers = []
    for scenario in scenarios:
        train_set = train_runs.get(scenario, set())
        if not train_set:
            continue
        # Stratified sample
        chosen = rng.sample(sorted(train_set), min(samples_per_scenario, len(train_set)))
        obs = pd.read_parquet(obs_root / f"{scenario}_obs.parquet")
        obs_by_run = obs.set_index("run_id")
        scenario_md = md[md["scenario"] == scenario].set_index("run_id")
        for rid in chosen:
            try:
                run_obs = obs_by_run.loc[[rid]] if rid in obs_by_run.index else None
            except KeyError:
                run_obs = None
            if run_obs is None or len(run_obs) == 0:
                continue
            run_obs = run_obs.sort_values("time_s")
            # Snapshot 30% through the run
            t_pick = float(run_obs["time_s"].iloc[int(0.3 * len(run_obs))])
            snap = run_obs.iloc[int(0.3 * len(run_obs))]
            try:
                row_md = scenario_md.loc[rid]
            except KeyError:
                continue

            outcome_stage = int(row_md.get("failure_stage_max", 0))
            duration_s    = float(row_md.get("run_duration_s", 0.0))
            sep_onset     = row_md.get("sep_onset_time_s", None)
            sep_onset = (None if sep_onset is None or pd.isna(sep_onset)
                         else float(sep_onset))

            peer = {
                "peer_run_id":  str(rid),
                "scenario":     scenario,
                "cause_family": _family_id(scenario),
                "snapshot_at_s": t_pick,
                "snapshot": {
                    "voltage_V":      round(float(snap.get("voltage_V_measured", 0)), 4),
                    "current_A":      round(float(snap.get("current_A_measured", 0)), 4),
                    "temperature_C":  round(float(snap.get("temperature_C_measured", 0)), 3),
                    "SOC":            round(float(snap.get("SOC_with_drift", 0)), 4),
                    "dT_dt_C_per_s":  round(float(snap.get("dT_dt_meas_C_per_s", 0)), 5),
                },
                "outcome": {
                    "failure_stage_max":  outcome_stage,
                    "did_escalate":       outcome_stage >= 2,
                    "reached_failure":    outcome_stage >= 3,
                    "duration_s":         duration_s,
                    "sep_onset_time_s":   sep_onset,
                },
            }
            # Deterministic hash placeholder — populated against PureChain in a
            # later phase (scripts/populate_hashes_v14.py)
            payload = json.dumps(peer, sort_keys=True).encode()
            peer["chain_tx_hash"] = hashlib.sha256(payload).hexdigest()
            peers.append(peer)
    return peers


def _family_id(scenario: str) -> int:
    for fid, members in CAUSE_FAMILIES.items():
        if scenario in members:
            return fid
    return -1


def _retrieve_peers(
    obs_snapshot: Dict[str, float],
    peer_corpus:  List[Dict[str, Any]],
    k: int = N_PEERS,
) -> List[Dict[str, Any]]:
    """
    k-nearest-neighbour retrieval over peer corpus, by Euclidean distance in a
    standardised observable feature space.  We don't use embeddings here:
    the feature set is small (5 channels) and the comparison is meaningful
    in physical units (after z-score normalisation).
    """
    if not peer_corpus:
        return []
    keys = ["voltage_V", "current_A", "temperature_C", "SOC", "dT_dt_C_per_s"]
    mat = np.array([[p["snapshot"][k] for k in keys] for p in peer_corpus], dtype=float)
    mu, sd = mat.mean(axis=0), mat.std(axis=0) + 1e-6
    norm = (mat - mu) / sd
    query = np.array([
        obs_snapshot.get("voltage_V_measured", 0.0),
        obs_snapshot.get("current_A_measured", 0.0),
        obs_snapshot.get("temperature_C_measured", 0.0),
        obs_snapshot.get("SOC_with_drift", 0.0),
        obs_snapshot.get("dT_dt_meas_C_per_s", 0.0),
    ])
    qn = (query - mu) / sd
    d  = np.linalg.norm(norm - qn, axis=1)
    order = d.argsort()[:k]
    return [peer_corpus[i] for i in order]


# ── Per-scenario sampling ─────────────────────────────────────

def _pick_decision_point(
    obs_run: pd.DataFrame,
    labels_run: pd.DataFrame,
    rng: random.Random,
    transition_bias: float = 0.5,
) -> int | None:
    """
    Pick t_decision such that:
      - t >= T_MIN (warm-up)
      - t <= run_duration - T_BUFFER_END (room for future labels)
      - with probability `transition_bias`, prefer timesteps that satisfy
        (failure_stage == 1 AND prevention_label_300s == 1) — the transient
        Stage-1-with-imminent-Stage-2 state that maps to GT=reorder_replacement
        and is otherwise rarely sampled by random selection.
    Returns the integer index into obs_run, or None if no valid point.
    """
    n = len(obs_run)
    if n < 10:
        return None
    obs_run    = obs_run.sort_values("time_s").reset_index(drop=True)
    labels_run = labels_run.sort_values("time_s").reset_index(drop=True)

    duration = float(obs_run["time_s"].iloc[-1])
    t_lo = T_MIN
    t_hi = duration - T_BUFFER_END
    if t_hi <= t_lo:
        return None

    valid = obs_run[(obs_run["time_s"] >= t_lo) & (obs_run["time_s"] <= t_hi)].index
    if len(valid) == 0:
        return None

    # Transition-bias: search labels for the (stage=1, p300=1) transient and
    # oversample it.  Use the labels' own time index since labels and obs are
    # already aligned at 5s sampling.
    if rng.random() < transition_bias:
        transition_mask = (
            (labels_run["failure_stage"] == 1)
            & (labels_run.get("prevention_label", 0) == 1)
            & (labels_run["time_s"] >= t_lo)
            & (labels_run["time_s"] <= t_hi)
        )
        transition_times = labels_run.loc[transition_mask, "time_s"].values
        if len(transition_times) > 0:
            t_target = float(rng.choice(transition_times))
            # Map back to obs_run index (nearest time_s)
            j = (obs_run["time_s"] - t_target).abs().idxmin()
            return int(j)

    return int(rng.choice(list(valid)))


# ── Main loader ────────────────────────────────────────────────

def load_v14_scenarios(
    n_total:      int           = 240,
    seed:         int           = 42,
    data_root:    Path          = V14_ROOT,
    families:     List[int]    | None = None,
    peer_samples_per_scenario: int  = 30,
) -> List[Dict[str, Any]]:
    """
    Build a stratified set of v14 scenario instances.

    Stratification: equal allocation across the 7 non-trivial cause families
    (excluding family 0, normal_cycle, which has no Stage>=2 transitions).
    """
    rng = random.Random(seed)
    if families is None:
        # Drop family 0 (normal_cycle) for stratified sampling — it contributes
        # only 'hold' GT actions and dilutes the action-class distribution.
        families = [1, 2, 3, 4, 5, 6, 7]

    md = pd.read_parquet(data_root / "run_metadata.parquet")
    split = json.load(open(data_root / "canonical_split.json"))
    train_runs = defaultdict(set)
    for sc, rid in split["train"]:
        train_runs[sc].add(rid)
    test_runs = defaultdict(set)
    for sc, rid in split["test"]:
        test_runs[sc].add(rid)

    # ── Build verified peer corpus from train split ──────────
    print(f"[v14_loader] Building peer corpus...")
    all_train_scenarios = sorted({
        sc for fid in CAUSE_FAMILIES for sc in CAUSE_FAMILIES[fid]
        if sc in train_runs and len(train_runs[sc]) > 0
    })
    peer_corpus = _build_peer_corpus(
        all_train_scenarios, train_runs, md, data_root,
        samples_per_scenario=peer_samples_per_scenario, rng=rng,
    )
    print(f"[v14_loader] Peer corpus: {len(peer_corpus)} entries")

    # ── Stratified sampling over (family, scenario) ───────────
    n_per_family = n_total // len(families)
    scenarios_out: List[Dict[str, Any]] = []

    for fid in families:
        family_scenarios = CAUSE_FAMILIES[fid]
        # Distribute n_per_family across scenarios in the family
        per_scenario = max(1, n_per_family // len(family_scenarios))
        for sc_name in family_scenarios:
            test_set = test_runs.get(sc_name, set())
            if not test_set:
                continue
            chosen = rng.sample(sorted(test_set), min(per_scenario, len(test_set)))
            obs_full    = pd.read_parquet(data_root / f"{sc_name}_obs.parquet")
            labels_full = pd.read_parquet(data_root / f"{sc_name}_labels.parquet")
            md_sc       = md[md["scenario"] == sc_name].set_index("run_id")

            for rid in chosen:
                obs_run = obs_full[obs_full["run_id"] == rid]
                lbl_run = labels_full[labels_full["run_id"] == rid]
                if len(obs_run) < 12 or len(lbl_run) < 12:
                    continue

                idx = _pick_decision_point(obs_run, lbl_run, rng)
                if idx is None:
                    continue

                # Window: 60s ending at t_decision
                obs_run = obs_run.sort_values("time_s").reset_index(drop=True)
                lbl_run = lbl_run.sort_values("time_s").reset_index(drop=True)
                t_dec = float(obs_run["time_s"].iloc[idx])
                t_w0  = t_dec - WINDOW_SECONDS
                window = obs_run[(obs_run["time_s"] >= t_w0) &
                                 (obs_run["time_s"] <= t_dec)]
                if len(window) < 6:
                    continue

                # Labels at decision point
                lbl_at = lbl_run[lbl_run["time_s"] == t_dec]
                if len(lbl_at) == 0:
                    # nearest
                    j = (lbl_run["time_s"] - t_dec).abs().idxmin()
                    lbl_at = lbl_run.loc[[j]]
                row = lbl_at.iloc[0]
                stage = int(row["failure_stage"])
                p300  = int(row.get("prevention_label", 0))
                p3600 = int(row.get("prevention_label_3600s", p300))
                gt    = _gt_action(stage, p300, p3600)

                # Observable snapshot at t_decision (used for peer retrieval)
                snap_row = obs_run.iloc[idx]
                snap = {ch: float(snap_row.get(ch, 0.0)) for ch in OBS_CHANNELS_PRIMARY}

                # Retrieve verified peers
                peers = _retrieve_peers(snap, peer_corpus, k=N_PEERS)

                # ── Assemble scenario dict ────────────────
                short = str(rid).zfill(6)[-6:]
                sid_seed = f"v14-{sc_name}-{rid}-{int(t_dec)}"
                sid = hashlib.sha256(sid_seed.encode()).hexdigest()[:32]

                summary = _summarise_window(window)
                try:
                    md_row = md_sc.loc[rid]
                    sensor_fault = bool(md_row.get("sensor_fault", False))
                except KeyError:
                    sensor_fault = False

                duration_s = float(obs_run["time_s"].iloc[-1])

                scenario_dict = {
                    "scenario_id":   sid,
                    "scenario_type": "v14_safety",
                    "asset_id":      f"B-{short}",
                    "query":         (
                        f"Battery asset B-{short} ({sc_name}) under monitoring at "
                        f"t={t_dec:.0f}s.  Recent observable channels show "
                        f"V={summary.get('voltage_V_measured',{}).get('last','?')}V, "
                        f"T={summary.get('temperature_C_measured',{}).get('last','?')}°C, "
                        f"trend={summary.get('temperature_C_measured',{}).get('trend','?')}.  "
                        f"Predict failure stage and decide an action."
                    ),
                    "ground_truth":  gt,
                    "label":         "valid_action",
                    "is_multi_constraint": False,
                    "metadata": {
                        "v14_run_id":            str(rid),
                        "v14_scenario":          sc_name,
                        "cause_family":          _family_id(sc_name),
                        "cause_family_name":     FAMILY_NAMES[_family_id(sc_name)],
                        "failure_stage":         stage,
                        "prevention_label_300s": p300,
                        "prevention_label_3600s":p3600,
                        "t_decision_s":          t_dec,
                        "run_duration_s":        duration_s,
                        "sensor_fault":          sensor_fault,
                    },
                    "obs_window": {
                        "t_start_s":   max(0.0, t_w0),
                        "t_end_s":     t_dec,
                        "n_timesteps": len(window),
                        "summary":     summary,
                        "snapshot":    {k: round(v, 5) for k, v in snap.items()},
                    },
                    "asset_records": peers,
                    "routing_records": [],
                }
                scenarios_out.append(scenario_dict)

    return scenarios_out


# ── CLI ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=240, help="total scenarios")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  type=Path, default=Path("data/synthetic/v14_scenarios.json"))
    ap.add_argument("--peers-per-scenario", type=int, default=30)
    args = ap.parse_args()

    scenarios = load_v14_scenarios(
        n_total=args.n, seed=args.seed,
        peer_samples_per_scenario=args.peers_per_scenario,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"\nWrote {len(scenarios)} v14 scenarios -> {args.out}")
    if scenarios:
        from collections import Counter
        gts = Counter(s["ground_truth"] for s in scenarios)
        cfs = Counter(s["metadata"]["cause_family_name"] for s in scenarios)
        print("Ground-truth action distribution:")
        for k, v in gts.most_common():
            print(f"  {k:<26} {v}")
        print("Cause-family distribution:")
        for k, v in cfs.most_common():
            print(f"  {k:<26} {v}")


if __name__ == "__main__":
    main()
