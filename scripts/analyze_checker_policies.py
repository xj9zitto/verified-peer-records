"""
analyze_checker_policies.py

Compares every physics-gate fallback policy on one set of model outputs.

The gated run stores the raw model text, so the parsed (stage, p300,
action) triple can be recovered and each policy replayed offline. All
policies therefore see identical model outputs and identical observable
windows, and differ only in what they do on a violation. That isolates
the fallback design from both the model's run-to-run variance and the
attack draw.

Policies compared:
  off               no repair (the model's own action)
  clamp             project the stage into the admissible interval
  defer             discard the output, use the rule engine
  clamp_then_defer  clamp, then defer if still inadmissible
  clamp_strict      clamp, but treating the weaker p300-support rule as
                    advisory rather than a violation

Output: results/checker_policy_comparison.json
"""

from __future__ import annotations
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.v14_attack import poison_obs_summary
from src.layer3.llm_agent_v14 import parse_v14_output
from src.layer3.v14_physics_checker import (
    check_v14, apply_fallback,
    POLICY_OFF, POLICY_CLAMP, POLICY_DEFER, POLICY_CLAMP_THEN_DEFER,
)

RESULTS = Path("results")
IN_FILE = RESULTS / "v14_poisoning_c4_gated_results.json"
SCEN_FILE = Path("data/synthetic/v14_scenarios.json")
OUT = RESULTS / "checker_policy_comparison.json"

ACTIONS = ["safe", "replace", "urgent"]
SEVERITY = {"safe": 0, "replace": 1, "urgent": 2}
ACTIONABLE = ("replace", "urgent")

# The p300-support rule is the weakest of the checks: it infers that an
# imminent-escalation claim is unsupported on a quiet window.  Treating
# it as advisory tests whether the gate's behaviour depends on it.
ADVISORY = {"p300_unsupported"}


def metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    conf = {t: Counter() for t in ACTIONS}
    for r in rows:
        conf[r["gt"]][r["pred"]] += 1
    acc = sum(1 for r in rows if r["pred"] == r["gt"]) / n
    act = [r for r in rows if r["gt"] in ACTIONABLE]
    act_acc = (sum(1 for r in act if r["pred"] == r["gt"]) / len(act)
               if act else None)
    under = sum(1 for r in rows
                if SEVERITY[r["pred"]] < SEVERITY[r["gt"]]) / n
    over = sum(1 for r in rows
               if SEVERITY[r["pred"]] > SEVERITY[r["gt"]]) / n
    per_class = {}
    for a in ACTIONS:
        sup = sum(conf[a].values())
        per_class[a] = (conf[a][a] / sup) if sup else None
    return {
        "n": n, "accuracy": acc, "actionable_accuracy": act_acc,
        "under_triage_rate": under, "over_triage_rate": over,
        "recall": per_class,
        "confusion": {t: dict(c) for t, c in conf.items()},
    }


def _load_records(args) -> List[Dict[str, Any]]:
    """Records for one condition, optionally taking the p = 0 outputs
    from another condition whose prompts are identical there (the
    round-2 re-run computes the nominal arm once)."""
    recs: List[Dict[str, Any]] = []
    for f in args.inputs:
        for r in json.loads(Path(f).read_text()):
            if r.get("classification") == "error":
                continue
            if r.get("attack_family", "standard") != "standard":
                continue
            if args.run_tag and r.get("run_tag", "") != args.run_tag:
                continue
            if r["condition"] == args.condition and r["rate"] > 0:
                recs.append(r)
            elif r["rate"] == 0 and r["condition"] == (args.nominal_from
                                                       or args.condition):
                recs.append(r)
    return recs


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+",
                    default=[str(IN_FILE)])
    ap.add_argument("--condition", default="C4")
    ap.add_argument("--nominal-from", default=None,
                    help="condition whose p=0 outputs stand in (e.g. C2)")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_path = Path(args.out)
    recs = _load_records(args)
    scen = {s["scenario_id"]: s for s in json.loads(SCEN_FILE.read_text())}
    print("replaying %d decisions" % len(recs))

    policies = [
        ("off", POLICY_OFF, False),
        ("clamp", POLICY_CLAMP, False),
        ("defer", POLICY_DEFER, False),
        ("clamp_then_defer", POLICY_CLAMP_THEN_DEFER, False),
        ("clamp_strict", POLICY_CLAMP, True),
    ]

    rows_by_policy: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name, _, _ in policies}
    fired = Counter()

    for r in recs:
        sc = scen.get(r["scenario_id"])
        if sc is None:
            continue
        summ = copy.deepcopy((sc.get("obs_window") or {}).get("summary", {}))
        if r.get("poisoned"):
            poison_obs_summary(summ)

        parsed = parse_v14_output(r.get("llm_output") or "")
        stage, p300 = parsed["predicted_stage"], parsed["predicted_p300"]
        action = parsed["action"] or "uncertain"

        chk = check_v14(stage, p300, action, summ)

        for name, policy, advisory in policies:
            c = chk
            if advisory:
                keep = [v for v in chk.violations if v not in ADVISORY]
                c = type(chk)(ok=not keep, violations=keep,
                              stage_lo=chk.stage_lo, stage_hi=chk.stage_hi)
            if not c.ok:
                fired[name] += 1
            out = apply_fallback(c, stage, p300, action, summ, policy=policy)
            pred = out["decision"]
            if pred not in SEVERITY:
                pred = "safe"      # an unusable output takes no action
            rows_by_policy[name].append(
                {"gt": r["ground_truth"], "pred": pred, "rate": r["rate"]})

    report: Dict[str, Any] = {"n": len(recs), "policies": {}}
    print("\n%-17s %6s %8s %9s %8s %8s %8s %8s"
          % ("policy", "fired", "acc", "act_acc", "safe_r", "repl_r",
             "urg_r", "under"))
    for name, _, _ in policies:
        rows = rows_by_policy[name]
        m = metrics(rows)
        m["gate_fired"] = fired[name]
        report["policies"][name] = m
        rc = m["recall"]
        print("%-17s %6d %8.3f %9.3f %8.3f %8.3f %8.3f %8.3f"
              % (name, fired[name], m["accuracy"], m["actionable_accuracy"],
                 rc["safe"] or 0, rc["replace"] or 0, rc["urgent"] or 0,
                 m["under_triage_rate"]))

    report["inputs"] = args.inputs
    report["condition"] = args.condition
    report["nominal_from"] = args.nominal_from
    out_path.write_text(json.dumps(report, indent=2))
    print("\nSaved -> %s" % out_path)


if __name__ == "__main__":
    main()
