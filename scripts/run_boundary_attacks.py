"""
run_boundary_attacks.py

Where does content-bound verification stop working?

The poisoning study attacks a retrieved peer by rewriting its outcome
fields.  That changes the canonical payload, so the committed digest no
longer matches and the predicate rejects the record.  Reviewer 2 pointed
out that this measures only the part of the attack surface the guarantee
covers, and asked for the complement: substitution of complete genuine
record-handle pairs, and withholding.  Section IV-D excludes selection
integrity and completeness by construction; this script measures what
that exclusion costs.

Three attack families are swept over the same scenarios, rates, seeds,
and arms, so the in-scope and out-of-scope attacks are directly
comparable:

  content_tamper        outcome fields rewritten under a genuine handle
                        (the in-scope attack; digest mismatch, T4 defends)
  genuine_substitution  a different genuine peer with its own genuine
                        handle (out of scope; nothing mismatches)
  withhold              severe-outcome peers dropped (out of scope; every
                        survivor verifies)

The decision arms come from src/layer1/v14_transfer.py, the same code the
LLM-free transfer study runs, so only the attack differs.

Output: results/v14_boundary_attacks_results.json
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

from src.layer1.v14_transfer import ARMS, CachedChain, decide
from src.layer2.purechain_client import PureChainClient
from src.dataset.v14_attack import DEFAULT_RATES, is_poisoned
from src.dataset.v14_boundary_attack import (
    benign_peer_pool, apply_genuine_substitution, apply_withhold,
)
# The in-scope attack comes from the dependency-free module, pinned
# byte-identical to the experiment runner by
# src/tests/test_v14_attack_parity.py.
from src.dataset.v14_attack import poison_scenario as _poison_scenario

SCEN_FILE = Path("data/synthetic/v14_scenarios.json")
OUT_FILE  = Path("results/v14_boundary_attacks_results.json")
FAMILIES  = ["content_tamper", "genuine_substitution", "withhold"]


def _apply(family: str, sc: Dict[str, Any], pool, seed: int) -> Dict[str, Any]:
    if family == "content_tamper":
        return _poison_scenario(sc, "forged_hash")
    if family == "genuine_substitution":
        return apply_genuine_substitution(sc, pool, seed)
    if family == "withhold":
        return apply_withhold(sc)
    raise ValueError(family)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(42, 62)),
                    help="attack seeds; 20 by default, matching the "
                         "multi-seed transfer study")
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--scenarios", type=Path, default=SCEN_FILE)
    args = ap.parse_args()

    scenarios = json.loads(args.scenarios.read_text())
    pool = benign_peer_pool(scenarios)
    print(f"benign genuine peer pool: {len(pool)} records")

    pc = PureChainClient()
    pc.connect()
    chain = CachedChain(pc)

    records: List[Dict[str, Any]] = []
    total = (len(scenarios) * len(args.rates) * len(ARMS)
             * len(args.seeds) * len(args.families))
    bar = tqdm(total=total, desc="boundary", unit="run", dynamic_ncols=True)

    for family in args.families:
      for seed in args.seeds:
        for rate in args.rates:
          for sc in scenarios:
            sid = sc["scenario_id"]
            poisoned = is_poisoned(sid, rate, seed=seed)
            s_use = (_apply(family, sc, pool, seed) if poisoned
                     else copy.deepcopy(sc))
            for arm in ARMS:
                bar.update(1)
                r = decide(arm, s_use, chain)
                # Only the fields the analysis and the per-arm diagnostics
                # need.  20 seeds x 3 families x 4 rates x 4 arms x 235
                # scenarios is 225,600 rows, and the constant columns
                # decide() fills in for schema compatibility with the LLM
                # runs would multiply the artifact for no information.
                records.append({
                    "attack_family":   family,
                    "seed":            seed,
                    "scenario_id":     sid,
                    "ground_truth":    sc.get("ground_truth", ""),
                    "rate":            rate,
                    "poisoned":        poisoned,
                    "condition":       arm,
                    "decision":        r["decision"],
                    "predicted_stage": r["predicted_stage"],
                    "tamper_alerts":   r["tamper_alerts"],
                    "peers_recovered": r["peers_recovered"],
                    "n_peers":         r["n_peers"],
                    "n_withheld":      s_use.get("_n_withheld", 0),
                })
    bar.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, separators=(",", ":")))
    print("\nSaved -> %s  (%d records, %d distinct chain queries)"
          % (args.out, len(records), chain.calls))

    print("\nActionable-subset accuracy, mean over seeds:")
    for family in args.families:
        print(f"\n  {family}")
        print("    %-6s " % "rate" + "  ".join("%8s" % a for a in ARMS))
        for rate in args.rates:
            row = []
            for arm in ARMS:
                rs = [r for r in records
                      if r["attack_family"] == family and r["rate"] == rate
                      and r["condition"] == arm
                      and r["ground_truth"] in ("replace", "urgent")]
                acc = (sum(1 for r in rs if r["decision"] == r["ground_truth"])
                       / len(rs)) if rs else float("nan")
                row.append("%8.3f" % acc)
            print("    %-6.2f " % rate + "  ".join(row))


if __name__ == "__main__":
    main()
