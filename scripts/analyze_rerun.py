"""
analyze_rerun.py — tables for the round-2 same-session agent re-run.

Reads results/rerun_2026-10/ (written by scripts/r2_campaign.sh) and
rebuilds the agent tables of the manuscript from runs made in one
inference session:

  nominal      Table IV   (p = 0; C2's outputs are shared by C3d, C3o
                           and C4, whose prompts are byte-identical when
                           nothing is rejected)
  safety       Table VI   (urgent recall, under-triage, expected cost)
  actionable   Table VII  (actionable accuracy by corruption rate)
  mcnemar      Table VIII (paired exact/chi2 McNemar, Holm over 24 tests)
  decomp       Table IX   (by attack type, poisoned actionable cases)
  seeds        Table V    (C4 - C2 per attack draw, pooled cluster CI)
  variance     inference variability vs attack variability vs
               cross-session drift
  boundary     agent under out-of-scope attacks
  ablation     Table XI   (true / state-only outcomes on C4, shuffled at p=0)
  expanded     Table XII  (independent 490-scenario set)

Identical prompts share one output.  A temperature-0 output is
reproducible for a fixed request history but not for a fixed prompt
(results/rerun_2026-10/inference_determinism.json), so two conditions
that build byte-identical prompts are given the same output instead of
two draws of the history-dependent noise:
  * C3d, C3o and C4 with no rejected record build C2's prompt on the
    same (possibly poisoned) scenario;
  * C3o after a detection re-runs on the untampered scenario, whose
    prompt is the nominal C2 prompt.
Contrasts between conditions then arise only where their evidence
differs.  The number of replaced records, and how many already agreed,
is reported under "sharing".

Output: results/rerun_2026-10/rerun_analysis.json
"""

from __future__ import annotations
import itertools
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compute_safety_metrics import (
    metrics_for, paired_bootstrap_delta, wilson,
)
from scripts.compute_mcnemar_corrected import mcnemar, holm
from scripts.compute_review_metrics import (
    metrics_from_records, always_x, peer_vote_action,
)

R = Path("results/rerun_2026-10")
ARCH = Path("results/archive/pre_redeploy_2026-10")
OUT = R / "rerun_analysis.json"
SCEN = Path("data/synthetic/v14_scenarios.json")
RATES = [0.0, 0.15, 0.30, 0.50]
ACTIONABLE = ("replace", "urgent")
CONDS = ["C0", "C1", "C2", "C3d", "C4", "C3o"]
CONTRASTS = [("C4", "C0"), ("C4", "C3d"), ("C4", "C1"), ("C4", "C2"),
             ("C3d", "C0"), ("C3d", "C2")]


def _load(name: str) -> List[Dict[str, Any]]:
    p = R / name
    return json.loads(p.read_text()) if p.exists() else []


def label(r: Dict[str, Any]) -> str:
    if r["condition"] == "C3":
        return "C3d" if r.get("c3_mode") == "deployable" else "C3o"
    return r["condition"]


def correct(r) -> int:
    return int(r["decision"] == r["ground_truth"])


def by(recs, **kw):
    return [r for r in recs if all(
        (label(r) if k == "cond" else r.get(k)) == v for k, v in kw.items())]


def acc(recs, actionable=False):
    rs = [r for r in recs if not actionable or r["ground_truth"] in ACTIONABLE]
    return round(sum(map(correct, rs)) / len(rs), 4) if rs else None


def index(recs, actionable=True) -> Dict[str, int]:
    return {r["scenario_id"]: correct(r) for r in recs
            if not actionable or r["ground_truth"] in ACTIONABLE}


def shared_nominal(nominal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """p = 0: C3d, C3o and C4 inherit C2's outputs (identical prompts)."""
    out = list(nominal)
    for r in by(nominal, cond="C2", rate=0.0):
        for cond in ("C3d", "C3o", "C4"):
            x = dict(r, shared_nominal=True)
            if cond == "C4":
                x["condition"] = "C4"
            else:
                x["condition"] = "C3"
                x["c3_mode"] = "deployable" if cond == "C3d" else "oracle"
            out.append(x)
    return out


SHARE_FIELDS = ("decision", "decision_raw", "predicted_stage",
                "predicted_p300", "classification", "llm_output")


def share_identical(recs: List[Dict[str, Any]],
                    nominal_c2: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Give identical-prompt records the C2 output (in place)."""
    c2 = {}
    for r in recs:
        if r["condition"] == "C2":
            c2[(r["scenario_id"], r["rate"], r.get("attack_seed"),
                r.get("run_tag", ""), r.get("attack_family", "standard"))] = r
    stats = Counter()
    for r in recs:
        cond = label(r)
        if cond not in ("C3d", "C3o", "C4") or r.get("shared_nominal"):
            continue
        if cond == "C3o" and r.get("tamper_alerts", 0) > 0:
            src = nominal_c2.get(r["scenario_id"])
            kind = "C3o_detected<-C2@0"
        elif r.get("tamper_alerts", 0) == 0:
            src = c2.get((r["scenario_id"], r["rate"], r.get("attack_seed"),
                          r.get("run_tag", ""),
                          r.get("attack_family", "standard")))
            kind = "%s_no_reject<-C2" % cond
        else:
            continue
        if src is None:
            stats[kind + "_missing"] += 1
            continue
        stats[kind] += 1
        stats[kind + "_already_same_decision"] += int(
            r["decision"] == src["decision"])
        for f in SHARE_FIELDS:
            r[f] = src.get(f)
        r["shared_from"] = kind
    return dict(stats)


def flips(a: List[Dict], b: List[Dict], actionable=False) -> Dict[str, Any]:
    da = {r["scenario_id"]: r["decision"] for r in a
          if not actionable or r["ground_truth"] in ACTIONABLE}
    db = {r["scenario_id"]: r["decision"] for r in b
          if not actionable or r["ground_truth"] in ACTIONABLE}
    keys = sorted(set(da) & set(db))
    n = sum(da[k] != db[k] for k in keys)
    return {"flipped": n, "of": len(keys),
            "rate": round(n / len(keys), 4) if keys else None}


def cluster_delta(a_by_seed: List[Dict[str, int]],
                  b_by_seed: List[Dict[str, int]], n_boot=10000, seed=42):
    """Pooled mean difference over seeds, bootstrapping scenarios."""
    keys = sorted(set.intersection(*[set(d) for d in a_by_seed + b_by_seed]))
    diffs = [st.mean(a[k] for a in a_by_seed) - st.mean(b[k] for b in b_by_seed)
             for k in keys]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return {"delta": round(st.mean(diffs), 4),
            "ci_low": round(boots[int(0.025 * n_boot)], 4),
            "ci_high": round(boots[int(0.975 * n_boot) - 1], 4),
            "n_scenarios": n, "n_obs": n * len(a_by_seed)}


def main() -> None:
    nominal = _load("nominal.json")
    main_runs = _load("main_c0124.json")
    c3 = _load("c3_both.json")
    rep1 = [r for r in shared_nominal(nominal) + main_runs + c3
            if r.get("run_tag", "rep1") == "rep1"]
    rep1 = [r for r in rep1 if r.get("attack_family", "standard") == "standard"]
    nominal_c2 = {r["scenario_id"]: r for r in nominal
                  if r["condition"] == "C2" and r["rate"] == 0.0}
    sharing = {"main": share_identical(rep1, nominal_c2)}
    errors = sum(r.get("decision") == "error" for r in rep1)
    failclosed = sum(r.get("decision_source") == "rule_engine_failclosed"
                     for r in rep1)
    rep = {"n_records": len(rep1), "errors": errors, "fail_closed": failclosed,
           "sharing": sharing}

    # -- Table IV: nominal ---------------------------------------
    scen = json.loads(SCEN.read_text())
    gts = [s["ground_truth"] for s in scen]
    t4 = {}
    for cond in ("C0", "C1", "C2"):
        rs = by(rep1, cond=cond, rate=0.0)
        m = metrics_from_records(rs)
        stage = [r for r in rs if r.get("predicted_stage") is not None
                 and r.get("true_failure_stage") is not None]
        p300 = [r for r in rs if r.get("predicted_p300") is not None
                and r.get("true_p300") is not None]
        t4[cond] = {
            "n": len(rs), "action_acc": round(m["global_acc"], 4),
            "stage_acc": round(sum(int(r["predicted_stage"]) == int(r["true_failure_stage"])
                                   for r in stage) / len(rs), 4) if rs else None,
            "p300_acc": round(sum(int(r["predicted_p300"]) == int(r["true_p300"])
                                  for r in p300) / len(rs), 4) if rs else None,
            "macro_f1": round(m["macro_f1"], 4),
            "missed_actionable": round(m["missed_actionable_rate"], 4),
            "false_intervention": round(m["false_intervention_rate"], 4),
            "per_class_recall": {c: (round(v["recall"], 4) if v["recall"] is not None else None)
                                 for c, v in m["per_class"].items()},
            "actionable_acc": acc(rs, True),
        }
    pv = [{"ground_truth": s["ground_truth"],
           "decision": peer_vote_action(s["asset_records"], False)} for s in scen]
    for name, m in (("peer_vote", metrics_from_records(pv)),
                    ("always_safe", always_x(gts, "safe")),
                    ("always_replace", always_x(gts, "replace"))):
        t4[name] = {"action_acc": round(m["global_acc"], 4),
                    "macro_f1": round(m["macro_f1"], 4),
                    "missed_actionable": round(m["missed_actionable_rate"], 4),
                    "false_intervention": round(m["false_intervention_rate"], 4),
                    "actionable_acc": round(m["actionable_acc"], 4)}
    rep["table_iv_nominal"] = t4

    # -- Table VI: safety ----------------------------------------
    t6 = {}
    for cond in CONDS:
        for rate in (0.0, 0.50):
            m = metrics_for(by(rep1, cond=cond, rate=rate))
            if not m.get("n"):
                continue
            t6["%s@%.2f" % (cond, rate)] = {
                "urgent_recall": round(m["urgent_recall"], 4),
                "under_triage": round(m["under_triage_rate"], 4),
                "cost_c100": round(m["expected_cost"]["c_miss_100"], 3),
                "cost_sweep": {k: round(v, 3) for k, v in m["expected_cost"].items()},
                "false_intervention": round(m["false_intervention_rate"], 4),
            }
    order = {}
    for ck in ("c_miss_10", "c_miss_50", "c_miss_100", "c_miss_500", "c_miss_1000"):
        order[ck] = sorted((c for c in CONDS if c != "C3o"),
                           key=lambda c: t6["%s@0.50" % c]["cost_sweep"][ck])
    rep["table_vi_safety"] = t6
    rep["cost_ordering_p50"] = order
    rep["cheapest_by_rate_c100"] = {
        "%.2f" % rate: min((c for c in CONDS if c != "C3o"),
                           key=lambda c: metrics_for(by(rep1, cond=c, rate=rate))
                           ["expected_cost"]["c_miss_100"])
        for rate in RATES}

    # -- Table VII: actionable accuracy --------------------------
    t7 = {}
    for rate in RATES:
        row = {}
        for cond in CONDS:
            rs = by(rep1, cond=cond, rate=rate)
            act = [r for r in rs if r["ground_truth"] in ACTIONABLE]
            k = sum(map(correct, act))
            row[cond] = {"acc": acc(rs, True), "n": len(act),
                         "wilson": wilson(k, len(act)),
                         "overall": acc(rs)}
        t7["%.2f" % rate] = row
    rep["table_vii_actionable"] = t7

    # -- Table VIII: McNemar -------------------------------------
    tests = []
    for rate in RATES:
        for a, b in CONTRASTS:
            ia = index(by(rep1, cond=a, rate=rate))
            ib = index(by(rep1, cond=b, rate=rate))
            keys = sorted(set(ia) & set(ib))
            bb = sum(ia[k] == 1 and ib[k] == 0 for k in keys)
            cc = sum(ia[k] == 0 and ib[k] == 1 for k in keys)
            t = mcnemar(bb, cc)
            t.update({"contrast": "%s-%s" % (a, b), "rate": rate,
                      "delta_pp": round(100 * (st.mean(ia[k] for k in keys)
                                               - st.mean(ib[k] for k in keys)), 1),
                      "n": len(keys)})
            boot = paired_bootstrap_delta(ia, ib)
            t["boot"] = {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in (boot or {}).items()}
            tests.append(t)
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = round(ph, 4)
        t["p"] = round(t["p"], 4)
    rep["table_viii_mcnemar"] = tests

    # -- Table IX: attack decomposition ---------------------------
    t9 = {}
    for rate in RATES[1:]:
        row = {}
        for at in ("fake_hash", "forged_hash"):
            for cond in ("C2", "C3d", "C4", "C3o"):
                rs = [r for r in by(rep1, cond=cond, rate=rate, attack_type=at)
                      if r["ground_truth"] in ACTIONABLE]
                row["%s/%s" % (cond, at)] = {"acc": acc(rs), "n": len(rs)}
        pois = {}
        for cond in ("C1", "C2", "C3d", "C4", "C3o"):
            rs = [r for r in by(rep1, cond=cond, rate=rate)
                  if r.get("poisoned") and r["ground_truth"] in ACTIONABLE]
            pois[cond] = {"acc": acc(rs), "n": len(rs)}
        row["poisoned_actionable"] = pois
        c4 = by(rep1, cond="C4", rate=rate)
        row["c4_detection"] = {
            "tamper_alerts": sum(r.get("tamper_alerts", 0) for r in c4),
            "recovered": sum(r.get("peers_recovered", 0) for r in c4)}
        t9["%.2f" % rate] = row
    rep["table_ix_decomp"] = t9

    # -- Table V: attack draws ------------------------------------
    seeds = {42: rep1, 43: _load("seed43.json"), 44: _load("seed44.json")}
    for sd in (43, 44):
        sharing["seed%d" % sd] = share_identical(seeds[sd], nominal_c2)
    t5 = {}
    for rate in RATES[1:]:
        per = {}
        a_s, b_s = [], []
        for sd, recs in seeds.items():
            ia = index(by(recs, cond="C4", rate=rate))
            ib = index(by(recs, cond="C2", rate=rate))
            if not ia or not ib:
                continue
            per[str(sd)] = round(st.mean(ia.values()) - st.mean(ib.values()), 4)
            a_s.append(ia)
            b_s.append(ib)
        row = {"per_seed": per}
        if len(a_s) > 1:
            row["pooled"] = cluster_delta(a_s, b_s)
        for cond in ("C0", "C2", "C4"):
            vals = [acc(by(recs, cond=cond, rate=rate), True)
                    for recs in seeds.values() if by(recs, cond=cond, rate=rate)]
            row["%s_acc_by_seed" % cond] = vals
            row["%s_sd_between_seeds" % cond] = (
                round(st.pstdev(vals), 4) if len(vals) > 1 else None)
        t5["%.2f" % rate] = row
    rep["table_v_seeds"] = t5

    # -- variance decomposition -----------------------------------
    var = {}
    vn = _load("variance_nominal.json")
    vp = _load("variance_p50.json")
    groups = {
        "C2@0.00": [by(rep1, cond="C2", rate=0.0)]
                   + [by(vn, run_tag=t) for t in ("rep2", "rep3")],
        "C2@0.50": [by(rep1, cond="C2", rate=0.5)]
                   + [by(vp, cond="C2", run_tag=t) for t in ("rep2", "rep3")],
        "C4@0.50": [by(rep1, cond="C4", rate=0.5)]
                   + [by(vp, cond="C4", run_tag=t) for t in ("rep2", "rep3")],
    }
    for g, reps in groups.items():
        reps = [x for x in reps if x]
        if len(reps) < 2:
            continue
        pairs = [flips(a, b) for a, b in itertools.combinations(reps, 2)]
        pairs_act = [flips(a, b, True) for a, b in itertools.combinations(reps, 2)]
        accs = [acc(x, True) for x in reps]
        var[g] = {"n_repeats": len(reps), "pairwise_flips": pairs,
                  "pairwise_flips_actionable": pairs_act,
                  "actionable_acc_by_repeat": accs,
                  "actionable_sd_inference": round(st.pstdev(accs), 4)}
    # identical prompts inside the main run: unpoisoned scenarios at p > 0
    same = defaultdict(lambda: [0, 0])
    for rate in RATES[1:]:
        d2 = {r["scenario_id"]: r for r in by(rep1, cond="C2", rate=rate)
              if not r.get("poisoned")}
        for cond in ("C3d", "C4"):
            for r in by(rep1, cond=cond, rate=rate):
                if r.get("poisoned") or r["scenario_id"] not in d2:
                    continue
                same[cond][1] += 1
                same[cond][0] += int(r.get("llm_output") ==
                                     d2[r["scenario_id"]].get("llm_output"))
    var["identical_prompts_unpoisoned_p_gt_0"] = {
        k: {"identical_outputs": v[0], "of": v[1]} for k, v in same.items()}
    # cross-session drift against the round-1 files (same prompts at p = 0
    # for C2 and C3, except the handle prefix shown for each peer)
    old = {}
    for name, f, cond, extra in (
            ("C2_may", "v14_poisoning_results.json", "C2", {}),
            ("C4_jul", "v14_poisoning_c4_results.json", "C4", {}),
            ("C3d_aug", "v14_poisoning_c3_deployable_results.json", "C3", {})):
        p = ARCH / f
        if not p.exists():
            continue
        recs = [r for r in json.loads(p.read_text())
                if r["condition"] == cond and r["rate"] == 0.0]
        old[name] = {"vs_new_C2_nominal": flips(by(rep1, cond="C2", rate=0.0), recs),
                     "vs_new_C2_nominal_actionable":
                         flips(by(rep1, cond="C2", rate=0.0), recs, True),
                     "old_actionable_acc": acc(recs, True)}
    var["cross_session_round1"] = old
    var["attack_sd_between_seeds"] = {
        k: {c: v.get("%s_sd_between_seeds" % c) for c in ("C2", "C4")}
        for k, v in t5.items()}
    rep["variance"] = var

    # -- boundary attacks at the agent -----------------------------
    bnd = {}
    for fam in ("genuine_substitution", "withhold"):
        recs = _load("boundary_%s.json" % fam)
        if not recs:
            continue
        sharing["boundary_" + fam] = share_identical(recs, nominal_c2)
        row = {}
        for cond in ("C0", "C2", "C4"):
            rs = by(recs, cond=cond, rate=0.5)
            pois = [r for r in rs if r.get("poisoned")
                    and r["ground_truth"] in ACTIONABLE]
            row[cond] = {"actionable_acc": acc(rs, True),
                         "poisoned_actionable_acc": acc(pois),
                         "n_poisoned_actionable": len(pois),
                         "urgent_recall": metrics_for(rs).get("urgent_recall"),
                         "tamper_alerts": sum(r.get("tamper_alerts", 0) for r in rs),
                         "mean_withheld": round(st.mean(
                             [r.get("n_withheld", 0) for r in rs if r.get("poisoned")]
                             or [0]), 3)}
        ia = index(by(recs, cond="C4", rate=0.5))
        ib = index(by(recs, cond="C2", rate=0.5))
        row["C4-C2"] = paired_bootstrap_delta(ia, ib)
        bnd[fam] = row
    std = {}
    for cond in ("C0", "C2", "C4"):
        rs = by(rep1, cond=cond, rate=0.5)
        pois = [r for r in rs if r.get("poisoned") and r["ground_truth"] in ACTIONABLE]
        std[cond] = {"actionable_acc": acc(rs, True),
                     "poisoned_actionable_acc": acc(pois)}
    bnd["content_tamper_standard"] = std
    rep["boundary_agent"] = bnd

    # -- Table XI: outcome ablations ------------------------------
    so = _load("ablation_stateonly.json")
    sh = _load("ablation_shuffled.json")
    t11 = {}
    for rate in RATES:
        true_c4 = by(rep1, cond="C4", rate=rate)
        st_rs = (by(so, cond="C2", rate=0.0) if rate == 0.0
                 else by(so, cond="C4", rate=rate))
        row = {"true_outcomes_C4": acc(true_c4, True),
               "state_only_C4": acc(st_rs, True)}
        if st_rs:
            m = metrics_for(st_rs)
            row["state_only_recall"] = {c: m["per_class"][c]["recall"]
                                        for c in ("safe", "replace", "urgent")}
            row["state_only_overall"] = acc(st_rs)
        mt = metrics_for(true_c4)
        row["true_recall"] = {c: mt["per_class"][c]["recall"]
                              for c in ("safe", "replace", "urgent")}
        row["true_overall"] = acc(true_c4)
        if rate == 0.0 and sh:
            row["shuffled_C2"] = acc(by(sh, cond="C2", rate=0.0), True)
            row["shuffled_overall"] = acc(by(sh, cond="C2", rate=0.0))
            row["shuffled_recall"] = {c: metrics_for(by(sh, cond="C2", rate=0.0))
                                      ["per_class"][c]["recall"]
                                      for c in ("safe", "replace", "urgent")}
        t11["%.2f" % rate] = row
    rep["table_xi_ablation"] = t11

    # -- Table XII: independent 490-scenario set -------------------
    ex = _load("expanded.json")
    if ex:
        ex_nom = {r["scenario_id"]: r for r in ex
                  if r["condition"] == "C2" and r["rate"] == 0.0}
        for r in [r for r in ex if r["condition"] == "C2" and r["rate"] == 0.0]:
            ex.append(dict(r, condition="C4", shared_nominal=True))
        sharing["expanded"] = share_identical(ex, ex_nom)
        t12 = {}
        for rate in RATES:
            ia = index(by(ex, cond="C4", rate=rate))
            ib = index(by(ex, cond="C2", rate=rate))
            keys = sorted(set(ia) & set(ib))
            bb = sum(ia[k] == 1 and ib[k] == 0 for k in keys)
            cc = sum(ia[k] == 0 and ib[k] == 1 for k in keys)
            boot = paired_bootstrap_delta(ia, ib)
            t12["%.2f" % rate] = {
                "C0": acc(by(ex, cond="C0", rate=rate), True),
                "C2": acc(by(ex, cond="C2", rate=rate), True),
                "C4": acc(by(ex, cond="C4", rate=rate), True),
                "n_actionable": len(keys),
                "delta": {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in (boot or {}).items()},
                "mcnemar": mcnemar(bb, cc),
                "under_triage_C2": round(metrics_for(by(ex, cond="C2", rate=rate))
                                         ["under_triage_rate"], 4),
                "under_triage_C4": round(metrics_for(by(ex, cond="C4", rate=rate))
                                         ["under_triage_rate"], 4),
                "urgent_recall_C2": metrics_for(by(ex, cond="C2", rate=rate))["urgent_recall"],
                "urgent_recall_C4": metrics_for(by(ex, cond="C4", rate=rate))["urgent_recall"],
            }
        rep["table_xii_expanded"] = t12

    # -- latency ---------------------------------------------------
    inf = sorted(r["latency_inference_ms"] for r in rep1
                 if r.get("decision_source") == "llm" and not r.get("shared_nominal")
                 and not r.get("shared_from") and r.get("latency_inference_ms"))
    rep["latency_inference_ms"] = {
        "n": len(inf), "median": round(st.median(inf), 1),
        "mean": round(st.mean(inf), 1), "p95": round(inf[int(0.95 * len(inf))], 1)}

    OUT.write_text(json.dumps(rep, indent=2))
    print("Saved -> %s  (%d records, %d errors, %d fail-closed)"
          % (OUT, len(rep1), errors, failclosed))
    print("\nActionable accuracy (Table VII):")
    print("  rate  " + "  ".join("%6s" % c for c in CONDS))
    for rate in RATES:
        row = t7["%.2f" % rate]
        print("  %.2f  " % rate + "  ".join(
            "%6.3f" % row[c]["acc"] if row[c]["acc"] is not None else "    NA"
            for c in CONDS))
    print("\nMcNemar at p=0.50:")
    for t in tests:
        if t["rate"] == 0.5:
            print("  %-8s b=%2d c=%2d  d=%+5.1fpp  p=%.4f  holm=%.3f"
                  % (t["contrast"], t["b"], t["c"], t["delta_pp"], t["p"], t["p_holm"]))


if __name__ == "__main__":
    main()
