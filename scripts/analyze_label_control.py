"""Verified-label control (R1, P1-4): C2 vs C2n.

C2n is C2 with every "verified"/PureChain claim in the prompt replaced by
neutral wording (llm_agent_v14.neutralise_labels); both conditions were
interleaved in one run (results/rerun_2026-10/label_control.json).
Writes results/rerun_2026-10/label_control_analysis.json.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_rerun import ACTIONABLE, acc, cluster_delta, flips, index  # noqa: E402

R = Path("results/rerun_2026-10")
IN = R / "label_control.json"
OUT = R / "label_control_analysis.json"


def mcnemar_exact(a, b):
    keys = sorted(set(a) & set(b))
    n01 = sum(a[k] == 1 and b[k] == 0 for k in keys)
    n10 = sum(a[k] == 0 and b[k] == 1 for k in keys)
    n = n01 + n10
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b": n01, "c": n10, "p": round(p, 4)}


def main():
    recs = json.loads(IN.read_text())
    main_c2 = {}
    for name in ("nominal.json", "main_c0124.json"):
        p = R / name
        if p.exists():
            for r in json.loads(p.read_text()):
                if r["condition"] == "C2" and r.get("attack_seed", 42) == 42:
                    main_c2.setdefault(r["rate"], []).append(r)
    rep = {"source": str(IN), "n_records": len(recs),
           "errors": sum(r["decision"] == "error" for r in recs),
           "by_rate": {}}
    for rate in (0.0, 0.5):
        c2 = [r for r in recs if r["condition"] == "C2" and r["rate"] == rate]
        c2n = [r for r in recs if r["condition"] == "C2n" and r["rate"] == rate]
        row = {
            "n": len(c2),
            "n_actionable": sum(r["ground_truth"] in ACTIONABLE for r in c2),
            "overall_acc": {"C2": acc(c2), "C2n": acc(c2n)},
            "actionable_acc": {"C2": acc(c2, True), "C2n": acc(c2n, True)},
            "delta_actionable_C2_minus_C2n":
                cluster_delta([index(c2)], [index(c2n)]),
            "delta_overall_C2_minus_C2n":
                cluster_delta([index(c2, False)], [index(c2n, False)]),
            "mcnemar_actionable": mcnemar_exact(index(c2), index(c2n)),
            "decision_flips_all": flips(c2, c2n),
            "decision_flips_actionable": flips(c2, c2n, True),
        }
        if rate > 0:
            pois = [r for r in c2 if r["poisoned"]
                    and r["ground_truth"] in ACTIONABLE]
            poisn = [r for r in c2n if r["poisoned"]
                     and r["ground_truth"] in ACTIONABLE]
            row["poisoned_actionable_acc"] = {"C2": acc(pois),
                                              "C2n": acc(poisn),
                                              "n": len(pois)}
        if rate in main_c2:
            row["history_noise_C2_vs_main_campaign_C2"] = {
                "flips_all": flips(c2, main_c2[rate]),
                "flips_actionable": flips(c2, main_c2[rate], True),
                "main_actionable_acc": acc(main_c2[rate], True)}
        rep["by_rate"]["%.2f" % rate] = row
    OUT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
