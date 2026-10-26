"""
run_predictor_fusion.py — a learned predictor consuming verified evidence.

Reviewer questions this answers (IoT-74062-2026, R1.2, R2.5): is the
language model necessary, and can the integrity layer be paired with a
stronger predictor consuming the same authenticated evidence?

A cost-weighted gradient-boosting classifier is trained on the 60-s
window summary (the 36 features of train_ml_baselines.py) plus six
features of the retrieved peer set:

    modal peer failure stage (ties upward), fraction escalated,
    fraction reaching failure, mean peer snapshot temperature,
    mean peer snapshot dT/dt, number of peers k

Training points are drawn from the train split of the current benchmark
generation exactly as in train_ml_baselines.py.  Their peers come from
a train-split peer corpus built the way the evaluation corpus was
(snapshot at 30% of a run, outcome from run metadata), retrieved by the
same standardised k-NN, with the query's own run excluded.

Augmentation, fixed before any evaluation: a fraction ``--aug`` of
training rows is duplicated with its window replaced by the window of a
random genuinely safe training point (label and peers unchanged), so the
model sees telemetry that looks nominal while the evidence does not.
This uses no knowledge of the attacker's values.  A further 30% of rows
see a random peer subset of size 0..4 so that withheld evidence is
in-distribution.  ``--aug 0`` trains without the window swap.

Evaluation reuses the agents' attack draw (seeded, per scenario) over
20 seeds, with the verification arms of the agent study:

    G0   window only (the cost-weighted baseline of Table XVII)
    G2   window + peers as retrieved (unverified)
    G3d  window + handle-verified peers, rejected ones withheld
    G4   window + content-verified peers, rejected ones reconstructed

``--attack adaptive`` extends the store attack to everything this
predictor reads from a peer: besides the outcome fields, each retrieved
peer's state snapshot is set to nominal values.  The standard attack
rewrites outcomes only, which suffices against the agent and the
deterministic rule (they read outcomes) but leaves a learned predictor a
clean signal in the peer states, retrieved with the true query.

Output: results/v14_predictor_fusion[_<tag>].json (records + analysis)
"""

from __future__ import annotations
import argparse
import copy
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.train_ml_baselines import (
    featurize, COST_WEIGHT, SEED, V14_ROOT, FAMILIES, RUNS_PER_SCENARIO,
)
from src.dataset.v14_loader import (
    CAUSE_FAMILIES, OBS_CHANNELS_PRIMARY, WINDOW_SECONDS,
    _summarise_window, _pick_decision_point, _load_run_metadata,
    _family_id,
)
from src.dataset.v14_attack import (
    DEFAULT_RATES, is_poisoned, attack_type, poison_scenario,
)
from src.layer1.v14_rule_engine import gt_action_from
from src.layer1.v14_transfer import CachedChain
from src.layer2.purechain_client import PureChainClient
from src.layer3.llm_agent_v14 import (
    _verify_peer_records, _verify_peer_records_content,
    _recover_peer_from_chain,
)

PEERS_PER_SCENARIO = 30          # as in the evaluation corpus
K = 5
SNAP_KEYS = ["voltage_V", "current_A", "temperature_C", "SOC",
             "dT_dt_C_per_s"]
OBS_SNAP = {"voltage_V": "voltage_V_measured",
            "current_A": "current_A_measured",
            "temperature_C": "temperature_C_measured",
            "SOC": "SOC_with_drift",
            "dT_dt_C_per_s": "dT_dt_meas_C_per_s"}
ARMS = ["G0", "G2", "G3d", "G4"]
ACTIONABLE = ("replace", "urgent")
SEVERITY = {"safe": 0, "replace": 1, "urgent": 2}
COST = {
    "safe":    {"safe": 0.0,   "replace": 1.0,  "urgent": 5.0},
    "replace": {"safe": 20.0,  "replace": 0.0,  "urgent": 1.0},
    "urgent":  {"safe": 100.0, "replace": 20.0, "urgent": 0.0},
}


# ── peer features ─────────────────────────────────────────────

def peer_features(peers: List[Dict[str, Any]]) -> np.ndarray:
    if not peers:
        return np.zeros(6)
    stages = [int((p.get("outcome") or {}).get("failure_stage_max", 0))
              for p in peers]
    cnt = Counter(stages)
    top = max(cnt.values())
    modal = max(s for s, c in cnt.items() if c == top)
    esc = np.mean([bool((p.get("outcome") or {}).get("did_escalate", False))
                   for p in peers])
    fail = np.mean([bool((p.get("outcome") or {}).get("reached_failure", False))
                    for p in peers])
    t = np.mean([float((p.get("snapshot") or {}).get("temperature_C", 0.0))
                 for p in peers])
    dt = np.mean([float((p.get("snapshot") or {}).get("dT_dt_C_per_s", 0.0))
                  for p in peers])
    return np.asarray([modal, esc, fail, t, dt, len(peers)], dtype=float)


# ── training data ─────────────────────────────────────────────

def build_training(rng_seed: int = SEED):
    """Decision points (window features, label, snapshot, run) and a
    train-split peer corpus, from one pass over the parquet files."""
    split = json.loads((V14_ROOT / "canonical_split.json").read_text())
    train_runs: Dict[str, set] = {}
    for sc, rid in split["train"]:
        train_runs.setdefault(sc, set()).add(rid)
    md = _load_run_metadata(V14_ROOT)
    rng = random.Random(rng_seed)            # same stream as the baseline
    prng = random.Random(rng_seed + 1)       # separate stream for peers

    points: List[Dict[str, Any]] = []
    corpus: List[Dict[str, Any]] = []
    scenarios = [sc for fid in FAMILIES for sc in CAUSE_FAMILIES[fid]]
    for sc in scenarios:
        obs_path = V14_ROOT / ("%s_obs.parquet" % sc)
        lab_path = V14_ROOT / ("%s_labels.parquet" % sc)
        if not obs_path.exists() or sc not in train_runs:
            continue
        ids = sorted(train_runs[sc])
        rng.shuffle(ids)
        dec_ids = set(ids[:RUNS_PER_SCENARIO])
        peer_ids = set(prng.sample(sorted(train_runs[sc]),
                                   min(PEERS_PER_SCENARIO, len(train_runs[sc]))))
        obs = pd.read_parquet(
            obs_path, columns=["run_id", "time_s"] + OBS_CHANNELS_PRIMARY)
        obs = obs[obs["run_id"].isin(dec_ids | peer_ids)]
        lab = pd.read_parquet(
            lab_path,
            columns=["run_id", "time_s", "failure_stage", "prevention_label"])
        lab = lab[lab["run_id"].isin(dec_ids)]
        smd = md[md["scenario"] == sc].set_index("run_id")

        for rid, obs_run in obs.groupby("run_id", sort=False):
            obs_run = obs_run.sort_values("time_s").reset_index(drop=True)
            if rid in peer_ids and rid in smd.index:
                j = int(0.3 * len(obs_run))
                snap = obs_run.iloc[j]
                row = smd.loc[rid]
                stage = int(row.get("failure_stage_max", 0))
                corpus.append({
                    "peer_run_id": str(rid), "scenario": sc,
                    "cause_family": _family_id(sc),
                    "snapshot": {k: float(snap[OBS_SNAP[k]]) for k in SNAP_KEYS},
                    "outcome": {"failure_stage_max": stage,
                                "did_escalate": stage >= 2,
                                "reached_failure": stage >= 3},
                })
            if rid not in dec_ids:
                continue
            lab_run = lab[lab["run_id"] == rid]
            if lab_run.empty:
                continue
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
            points.append({
                "x": f, "run": str(rid), "scenario": sc,
                "y": gt_action_from(int(row["failure_stage"]),
                                    int(row["prevention_label"]), 0),
                "snap": {k: float(obs_run[OBS_SNAP[k]].iloc[j])
                         for k in SNAP_KEYS},
            })
        print("  %-28s points %4d  corpus %4d" % (sc, len(points), len(corpus)))
        del obs, lab
    return points, corpus


def retrieve(points, corpus, k: int = K):
    """Standardised k-NN over the corpus, excluding the query's own run."""
    mat = np.array([[p["snapshot"][c] for c in SNAP_KEYS] for p in corpus])
    mu, sd = mat.mean(axis=0), mat.std(axis=0) + 1e-6
    norm = (mat - mu) / sd
    runs = np.array([(p["scenario"], p["peer_run_id"]) for p in corpus],
                    dtype=object)
    for pt in points:
        q = (np.array([pt["snap"][c] for c in SNAP_KEYS]) - mu) / sd
        order = np.linalg.norm(norm - q, axis=1).argsort()
        chosen = []
        for i in order:
            if runs[i][0] == pt["scenario"] and runs[i][1] == pt["run"]:
                continue                       # never the query's own run
            chosen.append(corpus[i])
            if len(chosen) == k:
                break
        pt["peers"] = chosen


def training_matrix(points, aug: float, seed: int):
    rng = random.Random(seed)
    safe_windows = [p["x"] for p in points if p["y"] == "safe"]
    X, y = [], []
    for p in points:
        peers = p["peers"]
        if rng.random() < 0.30:
            peers = rng.sample(peers, rng.randint(0, max(0, len(peers) - 1)))
        X.append(np.concatenate([p["x"], peer_features(peers)]))
        y.append(p["y"])
        if aug > 0 and rng.random() < aug:
            X.append(np.concatenate([rng.choice(safe_windows),
                                     peer_features(p["peers"])]))
            y.append(p["y"])
    return np.vstack(X), np.asarray(y)


def fit(X, y):
    m = HistGradientBoostingClassifier(random_state=SEED)
    m.fit(X, y, sample_weight=np.asarray([COST_WEIGHT[c] for c in y]))
    return m


# ── evaluation ────────────────────────────────────────────────

PEER_ROSY_SNAPSHOT = {"voltage_V": 3.65, "current_A": 0.45,
                      "temperature_C": 24.5, "SOC": 0.95,
                      "dT_dt_C_per_s": 0.001}


def adapt_attack(scenario):
    """Also falsify the state snapshot of every retrieved peer."""
    for p in scenario.get("asset_records", []) or []:
        p["snapshot"] = dict(PEER_ROSY_SNAPSHOT)
    return scenario


def peers_for_arm(arm, scenario, chain) -> Tuple[list, int, int]:
    peers = scenario.get("asset_records", []) or []
    if arm == "G0":
        return [], 0, 0
    if arm == "G2":
        return peers, 0, 0
    if arm == "G3d":
        use, n_rej = _verify_peer_records(peers, chain)
        return use, n_rej, 0
    if arm == "G4":
        use, rejected = _verify_peer_records_content(peers, chain)
        rec = 0
        for p in rejected:
            r = _recover_peer_from_chain(p, chain)
            if r is not None:
                use = use + [r]
                rec += 1
        return use, len(rejected), rec
    raise ValueError(arm)


def cluster_ci(a: Dict[str, List[int]], b: Dict[str, List[int]],
               n_boot: int = 10000, seed: int = SEED):
    keys = sorted(set(a) & set(b))
    diffs = [st.mean(a[k]) - st.mean(b[k]) for k in keys]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return {"delta": round(st.mean(diffs), 4),
            "ci_low": round(boots[int(0.025 * n_boot)], 4),
            "ci_high": round(boots[int(0.975 * n_boot) - 1], 4),
            "n_clusters": n}


def analyse(records, rates):
    out = {"by_arm": {}, "pairs": {}}
    for arm in ARMS:
        for rate in rates:
            rs = [r for r in records if r["arm"] == arm and r["rate"] == rate]
            act = [r for r in rs if r["gt"] in ACTIONABLE]
            urg = [r for r in rs if r["gt"] == "urgent"]
            per_seed = defaultdict(list)
            for r in act:
                per_seed[r["seed"]].append(r["pred"] == r["gt"])
            seed_acc = [st.mean(v) for v in per_seed.values()]
            out["by_arm"]["%s@%.2f" % (arm, rate)] = {
                "accuracy": round(st.mean(r["pred"] == r["gt"] for r in rs), 4),
                "actionable_accuracy": round(
                    st.mean(r["pred"] == r["gt"] for r in act), 4),
                "actionable_sd_between_seeds": round(
                    st.pstdev(seed_acc), 4) if len(seed_acc) > 1 else 0.0,
                "urgent_recall": round(
                    st.mean(r["pred"] == "urgent" for r in urg), 4),
                "under_triage": round(st.mean(
                    SEVERITY[r["pred"]] < SEVERITY[r["gt"]] for r in rs), 4),
                "cost": round(st.mean(COST[r["gt"]][r["pred"]] for r in rs), 3),
                "n_actionable": len(act),
            }
    for a, b in [("G4", "G2"), ("G4", "G3d"), ("G4", "G0"),
                 ("G3d", "G2"), ("G2", "G0")]:
        for rate in rates:
            pa, pb = defaultdict(list), defaultdict(list)
            for r in records:
                if r["rate"] != rate or r["gt"] not in ACTIONABLE:
                    continue
                if r["arm"] == a:
                    pa[r["sid"]].append(int(r["pred"] == r["gt"]))
                elif r["arm"] == b:
                    pb[r["sid"]].append(int(r["pred"] == r["gt"]))
            out["pairs"]["%s-%s@%.2f" % (a, b, rate)] = cluster_ci(pa, pb)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path,
                    default=Path("data/synthetic/v14_scenarios.json"))
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(42, 62)))
    ap.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--aug", type=float, default=0.30)
    ap.add_argument("--tag", default="")
    ap.add_argument("--attack", choices=["standard", "adaptive"],
                    default="standard")
    args = ap.parse_args()

    print("Building training points and the train-split peer corpus...")
    points, corpus = build_training()
    retrieve(points, corpus)
    print("  %d points, %d corpus peers, classes %s"
          % (len(points), len(corpus), dict(Counter(p["y"] for p in points))))

    Xw = np.vstack([p["x"] for p in points])
    yw = np.asarray([p["y"] for p in points])
    g0 = fit(Xw, yw)
    Xf, yf = training_matrix(points, args.aug, SEED)
    gf = fit(Xf, yf)
    print("  fitted G0 on %d rows, fused model on %d rows (aug=%.2f)"
          % (len(yw), len(yf), args.aug))

    scen = json.loads(args.scenarios.read_text())
    pc = PureChainClient()
    assert pc.connect(), "PureChain connection required for G3d/G4"
    chain = CachedChain(pc)

    records = []
    for seed in args.seeds:
        for rate in args.rates:
            rows, meta = [], []
            for sc in scen:
                sid = sc["scenario_id"]
                pois = is_poisoned(sid, rate, seed=seed)
                atype = attack_type(sid, args.alpha, seed=seed) if pois else "none"
                s_use = poison_scenario(sc, atype) if pois else copy.deepcopy(sc)
                if pois and args.attack == "adaptive":
                    adapt_attack(s_use)
                f = featurize((s_use.get("obs_window") or {}).get("summary", {}))
                if f is None:
                    continue
                for arm in ARMS:
                    use, n_rej, n_rec = peers_for_arm(arm, s_use, chain)
                    rows.append(np.concatenate([f, peer_features(use)]))
                    meta.append((sid, sc["ground_truth"], arm, pois, atype,
                                 n_rej, n_rec, len(use)))
            X = np.vstack(rows)
            pred_f = gf.predict(X)
            pred_0 = g0.predict(X[:, :Xw.shape[1]])
            for (sid, gt, arm, pois, atype, n_rej, n_rec, k), p0, pf in zip(
                    meta, pred_0, pred_f):
                records.append({"seed": seed, "rate": rate, "sid": sid,
                                "gt": gt, "arm": arm, "poisoned": pois,
                                "attack_type": atype, "tamper_alerts": n_rej,
                                "recovered": n_rec, "k": k,
                                "pred": str(p0 if arm == "G0" else pf)})
        print("  seed %d done (%d chain queries so far)" % (seed, chain.calls))

    analysis = analyse(records, args.rates)
    analysis["config"] = {"scenarios": str(args.scenarios),
                          "seeds": args.seeds, "alpha": args.alpha,
                          "aug": args.aug, "attack": args.attack,
                          "n_train_points": len(points),
                          "n_corpus": len(corpus),
                          "contract": pc.contract_address}
    tag = ("_" + args.tag) if args.tag else ""
    out = Path("results/v14_predictor_fusion%s.json" % tag)
    out.write_text(json.dumps({"analysis": analysis, "records": records},
                              separators=(",", ":")))
    print("\nSaved -> %s" % out)
    for rate in args.rates:
        print("  p=%.2f  " % rate + "  ".join(
            "%s act %.3f urg %.3f" % (
                arm, analysis["by_arm"]["%s@%.2f" % (arm, rate)]["actionable_accuracy"],
                analysis["by_arm"]["%s@%.2f" % (arm, rate)]["urgent_recall"])
            for arm in ARMS))
    for k, v in analysis["pairs"].items():
        if k.startswith("G4-G2") or k.startswith("G4-G0"):
            print("  %-14s %+.3f [%+.3f, %+.3f]"
                  % (k, v["delta"], v["ci_low"], v["ci_high"]))


if __name__ == "__main__":
    main()
