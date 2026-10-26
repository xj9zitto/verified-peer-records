"""
run_verification_transfer.py

Does the verification mechanism protect a decision architecture other
than the LLM agent?

A reviewer asked whether the observed robustness is attributable to the
proposed verification mechanism or to the specific LLM/RAG design.  This
experiment holds the attack and the verification code fixed and swaps the
decision architecture for a deterministic, non-LLM classifier
(``evidence_vote``) that consumes the same peer records.

Arms (chosen to mirror the LLM conditions exactly):

  T0  observables only, no peer evidence          (analogue of C0)
  T2  peers consumed as retrieved, unverified     (analogue of C2)
  T3  handle-existence check, rejected peers dropped   (analogue of C3)
  T4  content-bound verification + chain recovery (analogue of C4)

The poisoning draw reuses the LLM runner's seeded helpers, so scenario
selection and attack-type assignment are identical to the LLM arms and
the two architectures can be compared pair-for-pair.

Output: results/v14_verification_transfer_results.json  (same record
schema as the LLM runs, so compute_safety_metrics.py can consume it)
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.layer2.purechain_client import PureChainClient
# The arms and the chain cache moved to src/layer1/v14_transfer.py so that
# the boundary-attack study runs identical decision code; behaviour here is
# unchanged.
from src.layer1.v14_transfer import ARMS, CachedChain, decide as _decide
from src.dataset.v14_attack import (
    DEFAULT_RATES,
    is_poisoned as _is_poisoned,
    attack_type as _attack_type,
    poison_scenario as _poison_scenario,
)

SCEN_FILE = Path("data/synthetic/v14_scenarios.json")
OUT_FILE  = Path("results/v14_verification_transfer_results.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="attack seeds.  Each seed redraws which "
                         "scenarios are poisoned and which attack each "
                         "receives, giving independent attack instances "
                         "over the same scenario corpus.")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--scenarios", type=Path, default=SCEN_FILE)
    args = ap.parse_args()

    scenarios = json.loads(args.scenarios.read_text())
    pc = PureChainClient()
    pc.connect()
    chain = CachedChain(pc)

    records: List[Dict[str, Any]] = []
    total = (len(scenarios) * len(DEFAULT_RATES) * len(ARMS)
             * len(args.seeds))
    bar = tqdm(total=total, desc="transfer", unit="run", dynamic_ncols=True)

    for seed in args.seeds:
      for rate in DEFAULT_RATES:
        for sc in scenarios:
            sid = sc["scenario_id"]
            poisoned = _is_poisoned(sid, rate, seed=seed)
            atype = (_attack_type(sid, 0.5, seed=seed) if poisoned
                     else "none")
            s_use = (_poison_scenario(sc, atype) if poisoned
                     else copy.deepcopy(sc))

            for arm in ARMS:
                bar.update(1)
                r = _decide(arm, s_use, chain)
                records.append({
                    "seed":               seed,
                    "scenario_id":        sid,
                    "scenario_type":      sc.get("scenario_type", ""),
                    "ground_truth":       sc.get("ground_truth", ""),
                    "v14_scenario":       sc.get("metadata", {}).get("v14_scenario", ""),
                    "cause_family":       sc.get("metadata", {}).get("cause_family", -1),
                    "true_failure_stage": sc.get("metadata", {}).get("failure_stage"),
                    "true_p300":          sc.get("metadata", {}).get("prevention_label_300s"),
                    "rate":               rate,
                    "poisoned":           poisoned,
                    "attack_type":        atype,
                    **r,
                })
    bar.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))
    print("\nSaved -> %s  (%d records, %d distinct chain queries)"
          % (args.out, len(records), chain.calls))

    # Quick console summary on the actionable subset.
    print("\nActionable-subset accuracy (GT in {replace, urgent}):")
    print("  %-6s " % "rate" + "  ".join("%8s" % a for a in ARMS))
    for rate in DEFAULT_RATES:
        row = []
        for arm in ARMS:
            rs = [r for r in records
                  if r["rate"] == rate and r["condition"] == arm
                  and r["ground_truth"] in ("replace", "urgent")]
            acc = (sum(1 for r in rs if r["decision"] == r["ground_truth"])
                   / len(rs)) if rs else float("nan")
            row.append("%8.3f" % acc)
        print("  %-6.2f " % rate + "  ".join(row))


if __name__ == "__main__":
    main()
