"""
analyze_recovery_baseline.py

Summarises run_recovery_baseline.py into the small JSON the manuscript
and scripts/verify_paper_numbers.py cite.

The two claims that matter are agreement claims, not averages: the
signed-payload repository must reproduce the ledger arm decision for
decision, and the breached repository must reproduce the unverified and
handle-checked arms decision for decision.  Aggregate accuracy can
coincide while individual decisions differ, so both are counted
per paired decision rather than inferred from matching means.

Output: results/v14_recovery_baseline_analysis.json
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

IN_FILE = Path("results/v14_recovery_baseline_results.json")
OUT     = Path("results/v14_recovery_baseline_analysis.json")
ACTIONABLE = ("replace", "urgent")
ARMS = ["T0", "T2", "T3", "T4", "T5", "T5x"]
PAIRS = [("T5", "T4"), ("T5x", "T2"), ("T5x", "T3"), ("T4", "T2")]


def main() -> None:
    rows = json.loads(IN_FILE.read_text())
    rates = sorted({r["rate"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})

    idx: Dict[str, Dict[Any, Any]] = {}
    for r in rows:
        idx.setdefault(r["condition"], {})[
            (r["seed"], r["scenario_id"], r["rate"])] = r

    out: Dict[str, Any] = {
        "source": IN_FILE.name,
        "n_records": len(rows),
        "seeds": seeds,
        "note": ("T5 is a signature-authenticated payload repository with "
                 "an honest operator; T5x is the same repository with the "
                 "operator inside the same breach as the local store."),
        "by_rate": {},
        "agreement": {},
    }

    for rate in rates:
        per: Dict[str, Any] = {}
        for arm in ARMS:
            rs = [r for r in idx[arm].values() if r["rate"] == rate]
            act = [r for r in rs if r["ground_truth"] in ACTIONABLE]
            urg = [r for r in rs if r["ground_truth"] == "urgent"]
            per[arm] = {
                "accuracy": sum(r["decision"] == r["ground_truth"]
                                for r in rs) / len(rs),
                "actionable": sum(r["decision"] == r["ground_truth"]
                                  for r in act) / len(act),
                "urgent_recall": sum(r["decision"] == "urgent"
                                     for r in urg) / len(urg),
                "n_actionable": len(act),
            }
        out["by_rate"][f"{rate:.2f}"] = per

    for a, b in PAIRS:
        common = set(idx[a]) & set(idx[b])
        differ = sum(idx[a][k]["decision"] != idx[b][k]["decision"]
                     for k in common)
        out["agreement"][f"{a}-vs-{b}"] = {
            "n_paired": len(common),
            "n_differing_decisions": differ,
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {OUT}\n")
    print("  %-6s " % "rate" + "  ".join("%7s" % a for a in ARMS))
    for rate in rates:
        per = out["by_rate"][f"{rate:.2f}"]
        print("  %-6.2f " % rate
              + "  ".join("%7.3f" % per[a]["actionable"] for a in ARMS))
    print("\nPaired decision agreement:")
    for k, v in out["agreement"].items():
        print("  %-12s %d / %d decisions differ"
              % (k, v["n_differing_decisions"], v["n_paired"]))


if __name__ == "__main__":
    main()
