"""
train_temporal_baseline.py

Temporal prediction models for the battery decision task, completing the
baseline comparison a reviewer asked for (conventional classifiers,
cost-sensitive learning, and temporal models).

Why this runs on a different evaluation set from the summary-feature
baselines: the frozen 235-scenario evaluation set stores only the
per-channel window *summary*, not the raw sequence, and the trajectories
it was drawn from no longer exist in the regenerated benchmark. A
sequence model therefore cannot be scored on it. Instead every model
here -- temporal and summary-feature alike -- is trained on the
canonical train split and evaluated on a large held-out sample of the
canonical *test* split of the current generation, so the comparison
between representations is like for like.

Models:
  logreg   multinomial logistic regression on summary features
  gbm      histogram gradient boosting on summary features
  gru      single-layer GRU over the raw 12x6 window
  cnn      1-D convolutional network over the raw 12x6 window

Each is evaluated on honest input and under the same rosy-telemetry
falsification used against the agents, applied at sequence level by
flattening each channel to its falsified value.

Output: results/temporal_baselines.json
"""

from __future__ import annotations
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset.v14_attack import ROSY_TARGETS
from src.dataset.v14_loader import (
    CAUSE_FAMILIES, OBS_CHANNELS_PRIMARY, WINDOW_SECONDS,
    _pick_decision_point,
)
from src.layer1.v14_rule_engine import gt_action_from

V14_ROOT = Path("D:/VibeCodingProjects/synth-batt-data/data")
OUT = Path("results/temporal_baselines.json")

SEED = 42
FAMILIES = [1, 2, 3, 4, 5, 6, 7]
TRAIN_RUNS_PER_SCENARIO = 300
TEST_RUNS_PER_SCENARIO = 150
T_STEPS = 12                       # 60 s at 5 s resolution
ACTIONS = ["safe", "replace", "urgent"]
SEVERITY = {"safe": 0, "replace": 1, "urgent": 2}
ACTIONABLE = ("replace", "urgent")
COST_WEIGHT = {"safe": 5.0, "replace": 20.0, "urgent": 100.0}

torch.manual_seed(SEED)
np.random.seed(SEED)


def poison_window(w: np.ndarray) -> np.ndarray:
    """
    Apply the rosy telemetry attack at sequence level.

    The attack against the agents rewrites the window summary to benign
    constants.  The sequence-level equivalent is a flat window at those
    values, which reproduces the same falsified summary exactly.
    """
    out = w.copy()
    for j, ch in enumerate(OBS_CHANNELS_PRIMARY):
        tgt = ROSY_TARGETS.get(ch)
        if tgt:
            out[:, j] = tgt["mean"]
    return out


def build_split(runs: Dict[str, set], per_scenario: int,
                rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """Sample decision points and return (windows, actions)."""
    X: List[np.ndarray] = []
    y: List[str] = []
    scenarios = [sc for fid in FAMILIES for sc in CAUSE_FAMILIES[fid]]

    for sc in scenarios:
        obs_p = V14_ROOT / ("%s_obs.parquet" % sc)
        lab_p = V14_ROOT / ("%s_labels.parquet" % sc)
        if not obs_p.exists() or sc not in runs:
            continue
        ids = sorted(runs[sc])
        rng.shuffle(ids)
        ids = set(ids[:per_scenario])

        obs = pd.read_parquet(
            obs_p, columns=["run_id", "time_s"] + OBS_CHANNELS_PRIMARY)
        obs = obs[obs["run_id"].isin(ids)]
        lab = pd.read_parquet(
            lab_p,
            columns=["run_id", "time_s", "failure_stage", "prevention_label"])
        lab = lab[lab["run_id"].isin(ids)]

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
            # pad at the front / truncate from the front to T_STEPS
            if len(arr) < T_STEPS:
                arr = np.vstack([np.repeat(arr[:1], T_STEPS - len(arr), 0), arr])
            arr = arr[-T_STEPS:]
            row = lab_run.iloc[(lab_run["time_s"] - t_dec).abs().idxmin()]
            X.append(arr)
            y.append(gt_action_from(int(row["failure_stage"]),
                                    int(row["prevention_label"]), 0))
            n_add += 1
        print("  %-28s %4d" % (sc, n_add))
        del obs, lab

    return np.stack(X), np.asarray(y)


def summarize(W: np.ndarray) -> np.ndarray:
    """Summary features matching those used by the agent's prompt."""
    feats = [W.mean(1), W.min(1), W.max(1), W[:, 0, :], W[:, -1, :],
             np.sign(W[:, -1, :] - W[:, 0, :])]
    return np.concatenate(feats, axis=1)


class GRUNet(nn.Module):
    def __init__(self, n_ch: int, hidden: int = 48):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class CNNNet(nn.Module):
    def __init__(self, n_ch: int, width: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, width, 3, padding=1), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, 3))

    def forward(self, x):
        return self.net(x.transpose(1, 2))


def train_torch(model, Xtr, ytr, weights, epochs: int = 60):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss(reduction="none")
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    wt = torch.tensor(weights, dtype=torch.float32)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = (lossf(out, yt[idx]) * wt[idx]).mean()
            loss.backward()
            opt.step()
    return model


def metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    n = len(y_true)
    act = np.isin(y_true, ACTIONABLE)
    urg = y_true == "urgent"
    under = float(np.mean([SEVERITY[p] < SEVERITY[t]
                           for t, p in zip(y_true, pred)]))
    return {
        "n": n,
        "accuracy": float((pred == y_true).mean()),
        "actionable_accuracy": (float((pred[act] == y_true[act]).mean())
                                if act.any() else None),
        "urgent_recall": (float((pred[urg] == "urgent").mean())
                          if urg.any() else None),
        "under_triage_rate": under,
    }


def main() -> None:
    split = json.loads((V14_ROOT / "canonical_split.json").read_text())
    train_runs: Dict[str, set] = {}
    test_runs: Dict[str, set] = {}
    for sc, rid in split["train"]:
        train_runs.setdefault(sc, set()).add(rid)
    for sc, rid in split["test"]:
        test_runs.setdefault(sc, set()).add(rid)

    rng = random.Random(SEED)
    print("building train split...")
    Wtr, ytr = build_split(train_runs, TRAIN_RUNS_PER_SCENARIO, rng)
    print("building held-out test split...")
    Wte, yte = build_split(test_runs, TEST_RUNS_PER_SCENARIO, rng)
    print("train %s  test %s" % (Wtr.shape, Wte.shape))
    print("test class counts:", dict(Counter(yte)))

    # standardize per channel using train statistics
    mu = Wtr.reshape(-1, Wtr.shape[2]).mean(0)
    sd = Wtr.reshape(-1, Wtr.shape[2]).std(0) + 1e-8
    Wte_p = np.stack([poison_window(w) for w in Wte])

    def norm(W):
        return (W - mu) / sd

    Str, Ste, Ste_p = summarize(Wtr), summarize(Wte), summarize(Wte_p)
    classes = ACTIONS
    cls_idx = {c: i for i, c in enumerate(classes)}
    ytr_i = np.asarray([cls_idx[c] for c in ytr])
    weights = np.asarray([COST_WEIGHT[c] for c in ytr])

    report: Dict[str, Any] = {
        "n_train": int(len(Wtr)), "n_test": int(len(Wte)),
        "t_steps": T_STEPS, "channels": OBS_CHANNELS_PRIMARY,
        "test_class_counts": {k: int(v) for k, v in Counter(yte).items()},
        "models": {},
    }

    # -- summary-feature models -------------------------------
    sk_models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000,
                                                   random_state=SEED)),
        "gbm": HistGradientBoostingClassifier(random_state=SEED),
    }
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, m in sk_models.items():
        m.fit(Str, ytr, **({"sample_weight": weights}
                           if name == "gbm" else {}))
        preds[name] = (m.predict(Ste), m.predict(Ste_p))

    # -- temporal models --------------------------------------
    for name, ctor in (("gru", GRUNet), ("cnn", CNNNet)):
        model = train_torch(ctor(Wtr.shape[2]), norm(Wtr), ytr_i, weights)
        model.eval()
        with torch.no_grad():
            a = model(torch.tensor(norm(Wte), dtype=torch.float32)).argmax(1)
            b = model(torch.tensor(norm(Wte_p), dtype=torch.float32)).argmax(1)
        preds[name] = (np.asarray([classes[i] for i in a]),
                       np.asarray([classes[i] for i in b]))

    print("\n%-8s %-8s %8s %9s %9s %9s"
          % ("model", "input", "acc", "act_acc", "urg_rec", "under"))
    for name in ("logreg", "gbm", "gru", "cnn"):
        clean, pois = preds[name]
        mc, mp = metrics(yte, clean), metrics(yte, pois)
        report["models"][name] = {"clean": mc, "poisoned": mp}
        for lbl, m in (("honest", mc), ("poisoned", mp)):
            print("%-8s %-8s %8.3f %9.3f %9.3f %9.3f"
                  % (name, lbl, m["accuracy"], m["actionable_accuracy"],
                     m["urgent_recall"], m["under_triage_rate"]))
        print()

    OUT.write_text(json.dumps(report, indent=2))
    print("Saved -> %s" % OUT)


if __name__ == "__main__":
    main()
