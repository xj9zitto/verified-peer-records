"""
run_gbm_peer_fusion.py

Can the integrity layer be paired with a stronger predictor consuming
the same authenticated evidence?

Reviewer 2 asked this directly, and it is the same question the
Associate Editor raised about the role of the language model.  The
manuscript already shows two halves of the answer separately: gradient
boosting predicts far better than the agent on honest telemetry
(0.987 actionable) and collapses under a falsified window (0.494),
while content-bound verification protects the retrieved evidence
channel independently of what consumes it.  This script joins them.

The predictor is the cost-weighted gradient-boosting model of
train_ml_baselines.py, trained on the train split of the current
benchmark generation, which is disjoint from the 235 evaluated
scenarios by construction and by verification.  No new fitting happens
here, and nothing is tuned on the evaluation set: the peer evidence
enters through the same conservative rule the non-LLM arms already use,
taking whichever of the two readings is more severe.  The fusion is
therefore fixed in advance, and any gain belongs to the evidence, not
to a fitted combiner.

Arms mirror the agent conditions:

  G0  gradient boosting on the observable window only
  G2  plus unverified peers
  G3  plus handle-verified peers, rejected records withheld
  G4  plus content-bound verified peers with chain reconstruction

Output: results/v14_gbm_peer_fusion_results.json
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier

from src.layer1.v14_peer_vote import _peer_evidence
from src.layer1.v14_rule_engine import gt_action_from
from src.layer1.v14_transfer import CachedChain
from src.layer2.purechain_client import PureChainClient
from src.layer3.llm_agent_v14 import (
    _verify_peer_records, _verify_peer_records_content,
    _recover_peer_from_chain,
)
from src.dataset.v14_attack import (
    DEFAULT_RATES, is_poisoned, attack_type, poison_obs_summary,
)
from scripts.train_ml_baselines import (
    build_training_set, featurize, COST_WEIGHT, SEED,
)
from src.dataset.v14_attack import poison_scenario as _poison_scenario

SCEN_FILE = Path("data/synthetic/v14_scenarios.json")
OUT_FILE  = Path("results/v14_gbm_peer_fusion_results.json")
ARMS      = ["G0", "G2", "G3", "G4"]
SEVERITY  = {"safe": 0, "replace": 1, "urgent": 2}

# The cost matrix of compute_safety_metrics.py, reused unchanged.  A
# conservative fusion trades precision for recall, so accuracy alone
# cannot say whether the pairing is worth having; the paper's own
# argument is that it should be read against the cost of a missed
# runaway.
COST = {
    "safe":    {"safe": 0.0,   "replace": 1.0,  "urgent": 5.0},
    "replace": {"safe": 20.0,  "replace": 0.0,  "urgent": 1.0},
    "urgent":  {"safe": 100.0, "replace": 20.0, "urgent": 0.0},
}


def _peers_for_arm(arm: str, scenario: Dict[str, Any], chain
                   ) -> tuple[List[Dict[str, Any]], int, int]:
    peers = scenario.get("asset_records", []) or []
    if arm == "G0":
        return [], 0, 0
    if arm == "G2":
        return peers, 0, 0
    if arm == "G3":
        use, n_rej = _verify_peer_records(peers, chain)
        return use, n_rej, 0
    if arm == "G4":
        use, rejected = _verify_peer_records_content(peers, chain)
        recovered = 0
        for p in rejected:
            r = _recover_peer_from_chain(p, chain)
            if r is not None:
                use = use + [r]
                recovered += 1
        return use, len(rejected), recovered
    raise ValueError(arm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--scenarios", type=Path, default=SCEN_FILE)
    args = ap.parse_args()

    print("Fitting the cost-weighted gradient-boosting predictor on the "
          "train split...")
    Xtr, ytr, _ = build_training_set()
    gbm = HistGradientBoostingClassifier(random_state=SEED)
    gbm.fit(Xtr, ytr, sample_weight=np.asarray([COST_WEIGHT[c] for c in ytr]))
    print("  trained on %d points, %d features" % Xtr.shape)

    scen = json.loads(args.scenarios.read_text())
    pc = PureChainClient()
    pc.connect()
    chain = CachedChain(pc)

    records: List[Dict[str, Any]] = []
    total = len(scen) * len(args.rates) * len(ARMS) * len(args.seeds)
    bar = tqdm(total=total, desc="fusion", unit="run", dynamic_ncols=True)

    for seed in args.seeds:
      for rate in args.rates:
        for sc in scen:
            sid = sc["scenario_id"]
            poisoned = is_poisoned(sid, rate, seed=seed)
            atype = (attack_type(sid, args.alpha, seed=seed) if poisoned
                     else "none")
            s_use = (_poison_scenario(sc, atype) if poisoned
                     else copy.deepcopy(sc))

            summ = (s_use.get("obs_window") or {}).get("summary", {})
            f = featurize(summ)
            if f is None:
                bar.update(len(ARMS))
                continue
            gbm_action = str(gbm.predict(f.reshape(1, -1))[0])

            for arm in ARMS:
                bar.update(1)
                use, n_rej, n_rec = _peers_for_arm(arm, s_use, chain)
                stage_peer, p300_peer = _peer_evidence(use)
                peer_action = gt_action_from(stage_peer, p300_peer, 0)
                # Conservative fusion, fixed in advance: take whichever
                # reading is more severe.  Identical in form to the
                # observable floor the non-LLM arms apply.
                decision = max((gbm_action, peer_action),
                               key=lambda a: SEVERITY[a])
                records.append({
                    "seed":            seed,
                    "scenario_id":     sid,
                    "ground_truth":    sc.get("ground_truth", ""),
                    "rate":            rate,
                    "poisoned":        poisoned,
                    "attack_type":     atype,
                    "condition":       arm,
                    "decision":        decision,
                    "gbm_action":      gbm_action,
                    "peer_action":     peer_action,
                    "tamper_alerts":   n_rej,
                    "peers_recovered": n_rec,
                    "n_peers":         len(use),
                })
    bar.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, separators=(",", ":")))
    print("\nSaved -> %s  (%d records, %d distinct chain queries)"
          % (args.out, len(records), chain.calls))

    print("\n%-5s %-6s %8s %9s %9s %9s"
          % ("arm", "rate", "acc", "act_acc", "urg_rec", "cost"))
    for arm in ARMS:
        for rate in args.rates:
            rs = [r for r in records
                  if r["condition"] == arm and r["rate"] == rate]
            act = [r for r in rs if r["ground_truth"] in ("replace", "urgent")]
            urg = [r for r in rs if r["ground_truth"] == "urgent"]
            cost = sum(COST[r["ground_truth"]][r["decision"]]
                       for r in rs) / len(rs)
            print("%-5s %-6.2f %8.3f %9.3f %9.3f %9.2f"
                  % (arm, rate,
                     sum(r["decision"] == r["ground_truth"] for r in rs) / len(rs),
                     sum(r["decision"] == r["ground_truth"] for r in act) / len(act),
                     sum(r["decision"] == "urgent" for r in urg) / len(urg),
                     cost))
        print()


if __name__ == "__main__":
    main()
