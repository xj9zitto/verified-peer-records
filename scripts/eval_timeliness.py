"""
eval_timeliness.py

Sensitivity of warning timeliness to the sampling interval and to
communication delay, answering the part of the sampling/sensor comment
that the manuscript previously discussed but did not measure.

Two sweeps, both on the held-out test split of the current benchmark
generation (the frozen 235-scenario set stores only window summaries,
so no sequence-level resampling is possible on it):

  sampling   the 60-s window is decimated from its native 5-s
             resolution to 10, 20 and 30 s by keeping every k-th sample
             and carrying the last observation forward, which is what a
             slower BMS would actually deliver.  The model is unchanged;
             only the evidence it reads gets coarser.

  delay      the decision is formed on the window ending at t_d but
             takes effect delta seconds later, so it is scored against
             the ground-truth action at t_d + delta.  This measures how
             much of the 300-s prevention horizon a communication delay
             consumes.

1 s sampling is not evaluable: the benchmark is generated at 5 s, so a
finer interval would require inventing samples.

Output: results/timeliness_sensitivity.json
"""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier

from src.dataset.v14_loader import (
    CAUSE_FAMILIES, OBS_CHANNELS_PRIMARY, WINDOW_SECONDS,
    _pick_decision_point,
)
from src.layer1.v14_rule_engine import gt_action_from

V14_ROOT = Path("D:/VibeCodingProjects/synth-batt-data/data")
SPLIT = V14_ROOT / "canonical_split.json"
OUT = Path("results/timeliness_sensitivity.json")

SEED = 42
FAMILIES = [1, 2, 3, 4, 5, 6, 7]
TRAIN_PER_SCENARIO = 300
TEST_PER_SCENARIO = 150
T_STEPS = 12                       # 60 s at 5 s
ACTIONS = ["safe", "replace", "urgent"]
SEVERITY = {"safe": 0, "replace": 1, "urgent": 2}
ACTIONABLE = ("replace", "urgent")
COST_WEIGHT = {"safe": 5.0, "replace": 20.0, "urgent": 100.0}

SAMPLING_S = [5, 10, 20, 30]
DELAY_S = [0, 2, 5, 10, 30]


def load_split() -> Dict[str, Dict[str, set]]:
    """The split file lists (scenario, run_id) pairs per partition."""
    d = json.loads(SPLIT.read_text())
    out: Dict[str, Dict[str, set]] = {"train": {}, "test": {}}
    for part in ("train", "test"):
        for sc, rid in d[part]:
            out[part].setdefault(sc, set()).add(rid)
    return out


def build(runs: Dict[str, set], per_scenario: int, rng: random.Random,
          with_delays: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return windows, actions at t_d, and (optionally) actions at t_d+delta."""
    X: List[np.ndarray] = []
    y: List[str] = []
    yd: List[List[str]] = []
    scenarios = [sc for fid in FAMILIES for sc in CAUSE_FAMILIES[fid]]

    for sc in scenarios:
        obs_p = V14_ROOT / ("%s_obs.parquet" % sc)
        lab_p = V14_ROOT / ("%s_labels.parquet" % sc)
        if not obs_p.exists() or sc not in runs:
            continue
        ids = sorted(runs[sc])
        rng.shuffle(ids)
        keep = set(ids[:per_scenario])

        obs = pd.read_parquet(
            obs_p, columns=["run_id", "time_s"] + OBS_CHANNELS_PRIMARY)
        obs = obs[obs["run_id"].isin(keep)]
        lab = pd.read_parquet(
            lab_p,
            columns=["run_id", "time_s", "failure_stage", "prevention_label"])
        lab = lab[lab["run_id"].isin(keep)]

        n_add = 0
        for rid, obs_run in obs.groupby("run_id", sort=False):
            lab_run = lab[lab["run_id"] == rid]
            if lab_run.empty:
                continue
            obs_run = obs_run.sort_values("time_s").reset_index(drop=True)
            lab_run = lab_run.sort_values("time_s").reset_index(drop=True)
            j = _pick_decision_point(obs_run, lab_run, rng)
            if j is None:
                continue
            t_dec = float(obs_run["time_s"].iloc[j])
            win = obs_run[(obs_run["time_s"] > t_dec - WINDOW_SECONDS)
                          & (obs_run["time_s"] <= t_dec)]
            if len(win) < 3:
                continue
            arr = win[OBS_CHANNELS_PRIMARY].to_numpy(dtype=float)
            if len(arr) < T_STEPS:
                arr = np.vstack([np.repeat(arr[:1], T_STEPS - len(arr), 0), arr])
            arr = arr[-T_STEPS:]

            def action_at(t: float) -> str:
                row = lab_run.iloc[(lab_run["time_s"] - t).abs().idxmin()]
                return gt_action_from(int(row["failure_stage"]),
                                      int(row["prevention_label"]), 0)

            X.append(arr)
            y.append(action_at(t_dec))
            if with_delays:
                yd.append([action_at(t_dec + dl) for dl in DELAY_S])
            n_add += 1
        del obs, lab

    D = np.asarray(yd) if with_delays else np.empty((len(X), 0), dtype=object)
    return np.stack(X), np.asarray(y), D


def decimate(W: np.ndarray, every: int) -> np.ndarray:
    """Keep every `every`-th sample, carrying the last observation forward."""
    if every <= 1:
        return W
    out = W.copy()
    for t in range(T_STEPS):
        src = (t // every) * every
        out[:, t, :] = W[:, src, :]
    return out


def summarize(W: np.ndarray) -> np.ndarray:
    feats = [W.mean(1), W.min(1), W.max(1), W[:, 0, :], W[:, -1, :],
             np.sign(W[:, -1, :] - W[:, 0, :])]
    return np.concatenate(feats, axis=1)


def metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    act = np.isin(y_true, ACTIONABLE)
    urg = y_true == "urgent"
    under = sum(1 for t, p in zip(y_true, pred)
                if SEVERITY[p] < SEVERITY[t]) / len(y_true)
    cost = float(np.mean([0.0 if p == t else COST_WEIGHT[t]
                          for t, p in zip(y_true, pred)]))
    return {
        "n": int(len(y_true)),
        "accuracy": float((pred == y_true).mean()),
        "actionable_accuracy": float((pred[act] == y_true[act]).mean())
        if act.any() else None,
        "urgent_recall": float((pred[urg] == "urgent").mean())
        if urg.any() else None,
        "under_triage_rate": float(under),
        "expected_cost": cost,
    }


def main() -> None:
    rng = random.Random(SEED)
    split = load_split()

    print("building train split ...")
    Xtr, ytr, _ = build(split["train"], TRAIN_PER_SCENARIO, rng, False)
    print("building test split ...")
    Xte, yte, Yd = build(split["test"], TEST_PER_SCENARIO,
                         random.Random(SEED + 1), True)
    print("train %s   test %s" % (Xtr.shape, Xte.shape))

    w = np.array([COST_WEIGHT[a] for a in ytr])
    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=300)
    clf.fit(summarize(Xtr), ytr, sample_weight=w)

    report: Dict[str, Any] = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "note": "gradient boosting, cost-weighted; 5 s is the native rate",
        "sampling": {}, "delay": {},
    }

    print("\n-- sampling interval (window decimated, carry-forward) --")
    for s in SAMPLING_S:
        every = s // 5
        pred = clf.predict(summarize(decimate(Xte, every)))
        m = metrics(yte, pred)
        report["sampling"]["%ds" % s] = m
        print("   %-4s acc %.3f  act %.3f  urgent %.3f  under %.3f"
              % ("%ds" % s, m["accuracy"], m["actionable_accuracy"],
                 m["urgent_recall"], m["under_triage_rate"]))

    print("\n-- communication delay (decision scored at t_d + delta) --")
    pred0 = clf.predict(summarize(Xte))
    for i, dl in enumerate(DELAY_S):
        y_at = Yd[:, i]
        m = metrics(y_at, pred0)
        # how often the true action has become more severe by then
        worse = float(np.mean([SEVERITY[b] > SEVERITY[a]
                               for a, b in zip(yte, y_at)]))
        m["label_escalated_frac"] = worse
        report["delay"]["%ds" % dl] = m
        print("   %-4s acc %.3f  act %.3f  urgent %.3f  under %.3f  esc %.3f"
              % ("%ds" % dl, m["accuracy"], m["actionable_accuracy"],
                 m["urgent_recall"], m["under_triage_rate"], worse))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
