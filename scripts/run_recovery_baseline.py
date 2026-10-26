"""
run_recovery_baseline.py

Is reconstruction a property of the ledger, or of keeping an
authenticated copy of the payload?

Reviewer 2 observed that Table III compared a ledger that stores
canonical payloads against signatures and Merkle proofs that store
none, so "only the anchor reconstructs" credited the ledger with a
property that follows from payload retention.  A signature-based
repository holding authenticated payloads would reconstruct too.

This script measures that baseline against the on-chain path on the
same scenarios, rates, seeds, and attack, with the same decision rule,
so the only thing that varies is where the recovery copy lives and who
can take it away:

  T0   observables only
  T2   unverified peers
  T3   handle-existence check, rejected peers withheld
  T4   content-bound verification, recovery from the ledger
  T5   signature-authenticated repository, operator honest
  T5x  the same repository, operator inside the same breach as the
       local store, so the authenticated payload is withheld

The expected reading: T5 reproduces T4, which concedes the reviewer's
point that recovery is not exclusive to a ledger, while T5x collapses
toward T3, which is the property the ledger actually contributes --
the recovery copy is replicated by a validator set rather than served
by one operator the same adversary can reach.

Output: results/v14_recovery_baseline_results.json
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

from src.layer1.v14_transfer import CachedChain, decide
from src.layer2.purechain_client import PureChainClient
from src.layer2.v14_signed_repo import SignedRepository
from src.dataset.v14_attack import DEFAULT_RATES, is_poisoned, attack_type
from src.dataset.v14_attack import poison_scenario as _poison_scenario

SCEN_FILE = Path("data/synthetic/v14_scenarios.json")
OUT_FILE  = Path("results/v14_recovery_baseline_results.json")
ARMS      = ["T0", "T2", "T3", "T4", "T5", "T5x"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(42, 62)))
    ap.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="fabricated/valid-provenance mixture, as in the "
                         "headline poisoning run")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--scenarios", type=Path, default=SCEN_FILE)
    args = ap.parse_args()

    scenarios = json.loads(args.scenarios.read_text())
    repo = SignedRepository()
    print(f"signed repository: {len(repo.store)} payloads, "
          f"signer {repo.signer}")

    pc = PureChainClient()
    pc.connect()
    chain = CachedChain(pc)

    records: List[Dict[str, Any]] = []
    total = (len(scenarios) * len(args.rates) * len(args.arms)
             * len(args.seeds))
    bar = tqdm(total=total, desc="recovery", unit="run", dynamic_ncols=True)

    for seed in args.seeds:
      for rate in args.rates:
        for sc in scenarios:
            sid = sc["scenario_id"]
            poisoned = is_poisoned(sid, rate, seed=seed)
            atype = (attack_type(sid, args.alpha, seed=seed) if poisoned
                     else "none")
            s_use = (_poison_scenario(sc, atype) if poisoned
                     else copy.deepcopy(sc))
            for arm in args.arms:
                bar.update(1)
                r = decide(arm, s_use, chain, repo=repo)
                records.append({
                    "seed":            seed,
                    "scenario_id":     sid,
                    "ground_truth":    sc.get("ground_truth", ""),
                    "rate":            rate,
                    "poisoned":        poisoned,
                    "attack_type":     atype,
                    "condition":       arm,
                    "decision":        r["decision"],
                    "predicted_stage": r["predicted_stage"],
                    "tamper_alerts":   r["tamper_alerts"],
                    "peers_recovered": r["peers_recovered"],
                    "n_peers":         r["n_peers"],
                })
    bar.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, separators=(",", ":")))
    print("\nSaved -> %s  (%d records, %d distinct chain queries)"
          % (args.out, len(records), chain.calls))
    print("repository stats: %s" % repo.stats)

    print("\nActionable-subset accuracy, mean over seeds:")
    print("  %-6s " % "rate" + "  ".join("%7s" % a for a in args.arms))
    for rate in args.rates:
        row = []
        for arm in args.arms:
            rs = [r for r in records
                  if r["rate"] == rate and r["condition"] == arm
                  and r["ground_truth"] in ("replace", "urgent")]
            acc = (sum(1 for r in rs if r["decision"] == r["ground_truth"])
                   / len(rs)) if rs else float("nan")
            row.append("%7.3f" % acc)
        print("  %-6.2f " % rate + "  ".join(row))


if __name__ == "__main__":
    main()
