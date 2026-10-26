"""
eval_missing_measurements.py

Effect of missing measurements on the decision models, completing the
reviewer request that sensor noise, missing measurements, and
pack-level behaviour each be discussed and, where feasible, evaluated.

Two failure modes are modelled, both applied only at evaluation time so
the models never see them in training:

  dropout   a fraction of timesteps in the 60-s window is lost and
            carried forward from the last observation, the usual
            behaviour when a BMS frame is dropped
  channel   one observable channel is unavailable for the whole window
            and held at its training mean, i.e. a failed sensor

Reported against the same held-out decision points used for the
temporal baselines, so the degradation is comparable with the
falsified-telemetry column reported there.

Output: results/missing_measurements.json
"""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset.v14_loader import OBS_CHANNELS_PRIMARY
from scripts.train_temporal_baseline import (
    ACTIONS, COST_WEIGHT, SEED, TEST_RUNS_PER_SCENARIO,
    TRAIN_RUNS_PER_SCENARIO, V14_ROOT,
    CNNNet, GRUNet, build_split, metrics, summarize, train_torch,
)

OUT = Path("results/missing_measurements.json")
DROP_RATES = [0.0, 0.10, 0.25, 0.50]


def apply_dropout(W: np.ndarray, rate: float, rng: np.random.Generator
                  ) -> np.ndarray:
    """Lose a fraction of timesteps; hold the last observed value."""
    if rate <= 0:
        return W
    out = W.copy()
    n, t, c = out.shape
    lost = rng.random((n, t)) < rate
    lost[:, 0] = False          # the first sample anchors the carry-forward
    for i in range(n):
        for j in range(1, t):
            if lost[i, j]:
                out[i, j] = out[i, j - 1]
    return out


def apply_channel_loss(W: np.ndarray, ch_idx: int,
                       fill: np.ndarray) -> np.ndarray:
    """A sensor fails: the channel is held at its training mean."""
    out = W.copy()
    out[:, :, ch_idx] = fill[ch_idx]
    return out


def main() -> None:
    split = json.loads((V14_ROOT / "canonical_split.json").read_text())
    train_runs: Dict[str, set] = {}
    test_runs: Dict[str, set] = {}
    for sc, rid in split["train"]:
        train_runs.setdefault(sc, set()).add(rid)
    for sc, rid in split["test"]:
        test_runs.setdefault(sc, set()).add(rid)

    rng = random.Random(SEED)
    print("building splits...")
    Wtr, ytr = build_split(train_runs, TRAIN_RUNS_PER_SCENARIO, rng)
    Wte, yte = build_split(test_runs, TEST_RUNS_PER_SCENARIO, rng)
    print("train %s  test %s" % (Wtr.shape, Wte.shape))

    mu = Wtr.reshape(-1, Wtr.shape[2]).mean(0)
    sd = Wtr.reshape(-1, Wtr.shape[2]).std(0) + 1e-8

    def norm(W):
        return (W - mu) / sd

    weights = np.asarray([COST_WEIGHT[c] for c in ytr])
    cls_idx = {c: i for i, c in enumerate(ACTIONS)}
    ytr_i = np.asarray([cls_idx[c] for c in ytr])

    print("fitting models...")
    logreg = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000,
                                              random_state=SEED))
    logreg.fit(summarize(Wtr), ytr)
    gbm = HistGradientBoostingClassifier(random_state=SEED)
    gbm.fit(summarize(Wtr), ytr, sample_weight=weights)
    gru = train_torch(GRUNet(Wtr.shape[2]), norm(Wtr), ytr_i, weights)
    cnn = train_torch(CNNNet(Wtr.shape[2]), norm(Wtr), ytr_i, weights)
    gru.eval()
    cnn.eval()

    def predict(W: np.ndarray) -> Dict[str, np.ndarray]:
        S = summarize(W)
        with torch.no_grad():
            X = torch.tensor(norm(W), dtype=torch.float32)
            g = gru(X).argmax(1).numpy()
            c = cnn(X).argmax(1).numpy()
        return {
            "logreg": logreg.predict(S),
            "gbm": gbm.predict(S),
            "gru": np.asarray([ACTIONS[i] for i in g]),
            "cnn": np.asarray([ACTIONS[i] for i in c]),
        }

    report: Dict[str, Any] = {"n_test": int(len(Wte)),
                              "drop_rates": DROP_RATES,
                              "timestep_dropout": {}, "channel_loss": {}}

    nprng = np.random.default_rng(SEED)
    print("\n-- timestep dropout --")
    print("%-8s %-6s %8s %9s %9s %9s"
          % ("model", "drop", "acc", "act_acc", "urg_rec", "under"))
    for rate in DROP_RATES:
        Wd = apply_dropout(Wte, rate, nprng)
        for name, pred in predict(Wd).items():
            m = metrics(yte, pred)
            report["timestep_dropout"].setdefault(name, {})[
                "%.2f" % rate] = m
            print("%-8s %-6.2f %8.3f %9.3f %9.3f %9.3f"
                  % (name, rate, m["accuracy"], m["actionable_accuracy"],
                     m["urgent_recall"], m["under_triage_rate"]))
        print()

    print("-- single channel unavailable --")
    print("%-8s %-24s %8s %9s %9s"
          % ("model", "channel", "acc", "act_acc", "urg_rec"))
    for ci, ch in enumerate(OBS_CHANNELS_PRIMARY):
        Wc = apply_channel_loss(Wte, ci, mu)
        for name, pred in predict(Wc).items():
            m = metrics(yte, pred)
            report["channel_loss"].setdefault(name, {})[ch] = m
            if name == "gbm":
                print("%-8s %-24s %8.3f %9.3f %9.3f"
                      % (name, ch, m["accuracy"],
                         m["actionable_accuracy"], m["urgent_recall"]))

    OUT.write_text(json.dumps(report, indent=2))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
