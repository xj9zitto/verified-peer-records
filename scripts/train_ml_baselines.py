"""
train_ml_baselines.py

Conventional supervised baselines for the battery decision task, as a
reviewer asked: do standard classifiers on the same observable window
reach decision quality the LLM agent does not, and does the adversarial
picture change when the predictor changes?

Three baselines are trained on decision points drawn from the *train*
split and evaluated on the same 235 test decision points the agents see:

  logreg    multinomial logistic regression on standardized features
  hgb       histogram gradient boosting (the sklearn analogue of the
            gradient-boosted-tree baselines used in battery ML)
  hgb_cost  the same model with cost-sensitive sample weights, each
            sample weighted by the worst-case cost of misclassifying it

Features are the identical 60-s window summary the agents receive: per
channel the mean, min, max, first, last, and a trend sign.  No latent
or label-tier channel is used.  Evaluation repeats under the same
seeded poisoning draw as the agent runs, so the falsified-telemetry
attack applies to the classifiers exactly as it does to the agents.

Output: results/ml_baselines.json
"""

from __future__ import annotations
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset.v14_loader import (
    CAUSE_FAMILIES, OBS_CHANNELS_PRIMARY, WINDOW_SECONDS,
    _summarise_window, _pick_decision_point,
)
from src.layer1.v14_rule_engine import gt_action_from
from src.dataset.v14_attack import (
    is_poisoned as _is_poisoned,
    poison_obs_summary as _poison_obs_summary,
    DEFAULT_RATES,
)

V14_ROOT  = Path("D:/VibeCodingProjects/synth-batt-data/data")
SCEN_FILE = Path("data/synthetic/v14_scenarios.json")
OUT       = Path("results/ml_baselines.json")

SEED       = 42
FAMILIES   = [1, 2, 3, 4, 5, 6, 7]      # family 0 excluded, as in the loader
RUNS_PER_SCENARIO = 300
ACTIONS    = ["safe", "replace", "urgent"]
SEVERITY   = {"safe": 0, "replace": 1, "urgent": 2}
ACTIONABLE = ("replace", "urgent")
TREND      = {"rising": 1.0, "flat": 0.0, "falling": -1.0}
STATS      = ["mean", "min", "max", "first", "last"]

# Worst-case misclassification cost per true class (see
# compute_safety_metrics.py for the matrix these come from).
COST_WEIGHT = {"safe": 5.0, "replace": 20.0, "urgent": 100.0}


def featurize(summary: Dict[str, Any]) -> Optional[np.ndarray]:
    """Flatten a window summary into a fixed-length vector."""
    vec: List[float] = []
    for ch in OBS_CHANNELS_PRIMARY:
        s = summary.get(ch)
        if not s:
            return None
        for k in STATS:
            vec.append(float(s[k]))
        vec.append(TREND.get(s.get("trend", "flat"), 0.0))
    return np.asarray(vec, dtype=float)


def build_training_set() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Draw decision points from train-split runs and featurize them."""
    split = json.loads((V14_ROOT / "canonical_split.json").read_text())
    train_runs: Dict[str, set] = {}
    for sc, rid in split["train"]:
        train_runs.setdefault(sc, set()).add(rid)

    rng = random.Random(SEED)
    X: List[np.ndarray] = []
    y: List[str] = []

    scenarios = [sc for fid in FAMILIES for sc in CAUSE_FAMILIES[fid]]
    for sc in scenarios:
        obs_path = V14_ROOT / ("%s_obs.parquet" % sc)
        lab_path = V14_ROOT / ("%s_labels.parquet" % sc)
        if not obs_path.exists() or sc not in train_runs:
            print("  skip %s (missing)" % sc)
            continue

        ids = sorted(train_runs[sc])
        rng.shuffle(ids)
        ids = set(ids[:RUNS_PER_SCENARIO])

        obs = pd.read_parquet(
            obs_path, columns=["run_id", "time_s"] + OBS_CHANNELS_PRIMARY)
        obs = obs[obs["run_id"].isin(ids)]
        lab = pd.read_parquet(
            lab_path,
            columns=["run_id", "time_s", "failure_stage", "prevention_label"])
        lab = lab[lab["run_id"].isin(ids)]

        n_added = 0
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
            f = featurize(_summarise_window(win))
            if f is None:
                continue

            row = lab_run.iloc[(lab_run["time_s"] - t_dec).abs().idxmin()]
            stage = int(row["failure_stage"])
            p300 = int(row["prevention_label"])
            X.append(f)
            y.append(gt_action_from(stage, p300, 0))
            n_added += 1

        print("  %-28s %4d samples" % (sc, n_added))
        del obs, lab

    return np.vstack(X), np.asarray(y), scenarios


def evaluate(name: str, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    pred = model.predict(X)
    n = len(y)
    conf = {t: Counter() for t in ACTIONS}
    for t, p in zip(y, pred):
        conf[t][p] += 1
    acc = float((pred == y).mean())
    act_idx = np.isin(y, ACTIONABLE)
    act_acc = (float((pred[act_idx] == y[act_idx]).mean())
               if act_idx.any() else None)
    urg_idx = y == "urgent"
    urg_rec = (float((pred[urg_idx] == "urgent").mean())
               if urg_idx.any() else None)
    under = float(np.mean([SEVERITY[p] < SEVERITY[t]
                           for t, p in zip(y, pred)]))
    return {
        "model": name, "n": n, "accuracy": acc,
        "actionable_accuracy": act_acc, "urgent_recall": urg_rec,
        "under_triage_rate": under,
        "confusion": {t: dict(c) for t, c in conf.items()},
    }


def main() -> None:
    print("Building training set from the train split...")
    Xtr, ytr, _ = build_training_set()
    print("train: %d samples, %d features, classes %s"
          % (Xtr.shape[0], Xtr.shape[1], dict(Counter(ytr))))

    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, multi_class="multinomial",
                               random_state=SEED)),
        "hgb": HistGradientBoostingClassifier(random_state=SEED),
        "hgb_cost": HistGradientBoostingClassifier(random_state=SEED),
    }
    weights = np.asarray([COST_WEIGHT[c] for c in ytr])
    for name, m in models.items():
        if name == "hgb_cost":
            m.fit(Xtr, ytr, sample_weight=weights)
        else:
            m.fit(Xtr, ytr)
        print("  fitted %s" % name)

    # -- test set: the same 235 decision points the agents see -----
    scen = json.loads(SCEN_FILE.read_text())
    report: Dict[str, Any] = {
        "n_train": int(Xtr.shape[0]), "n_features": int(Xtr.shape[1]),
        "train_class_counts": {k: int(v) for k, v in Counter(ytr).items()},
        "runs_per_scenario": RUNS_PER_SCENARIO,
        "by_rate": {},
    }

    print("\n%-10s %-5s %8s %9s %9s %9s"
          % ("model", "rate", "acc", "act_acc", "urg_rec", "under"))
    for rate in DEFAULT_RATES:
        Xte, yte = [], []
        for sc in scen:
            summ = copy.deepcopy(
                (sc.get("obs_window") or {}).get("summary", {}))
            if _is_poisoned(sc["scenario_id"], rate):
                _poison_obs_summary(summ)
            f = featurize(summ)
            if f is None:
                continue
            Xte.append(f)
            yte.append(sc["ground_truth"])
        Xte = np.vstack(Xte)
        yte = np.asarray(yte)

        for name, m in models.items():
            r = evaluate(name, m, Xte, yte)
            report["by_rate"]["%s@%.2f" % (name, rate)] = r
            print("%-10s %-5.2f %8.3f %9s %9s %9.3f"
                  % (name, rate, r["accuracy"],
                     "%.3f" % r["actionable_accuracy"],
                     "%.3f" % r["urgent_recall"], r["under_triage_rate"]))
        print()

    OUT.write_text(json.dumps(report, indent=2))
    print("Saved -> %s" % OUT)


if __name__ == "__main__":
    main()
