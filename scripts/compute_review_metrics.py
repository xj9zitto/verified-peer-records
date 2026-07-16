"""
compute_review_metrics.py

Class-balanced diagnostics, trivial baselines, deterministic peer-vote
baseline, Holm-adjusted McNemar p-values, v14-task latency, and
auxiliary quantities requested in the IoT-J pre-submission review
(IOTJ_REVIEW_RESPONSE_PLAN.md, Phase B).

All quantities are computed from existing result artifacts; no LLM
calls are made.

Output: results/review_metrics.json + printed report.
"""

import json
from collections import Counter
from pathlib import Path

RESULTS = Path("results")
DATA    = Path("data/synthetic")

CLASSES = ["safe", "replace", "urgent"]
CONDS   = ["C0", "C1", "C2", "C3"]
ACTIONABLE = ("replace", "urgent")


# ── metric helpers ────────────────────────────────────────────────

def confusion(records):
    """3x3 confusion matrix over CLASSES + count of invalid predictions."""
    cm = {gt: {pr: 0 for pr in CLASSES} for gt in CLASSES}
    invalid = 0
    for r in records:
        gt, pr = r["ground_truth"], r["decision"]
        if gt not in CLASSES:
            continue
        if pr in CLASSES:
            cm[gt][pr] += 1
        else:
            invalid += 1  # counts against recall of gt, excluded from precision denominators
    return cm, invalid


def metrics_from_records(records):
    cm, invalid = confusion(records)
    n = sum(cm[g][p] for g in CLASSES for p in CLASSES) + invalid
    correct = sum(cm[c][c] for c in CLASSES)
    out = {"n": n, "invalid_outputs": invalid,
           "global_acc": correct / n if n else None,
           "confusion": cm}
    recalls, f1s = [], []
    per_class = {}
    for c in CLASSES:
        gt_n = sum(cm[c].values()) + (invalid if False else 0)
        gt_total = sum(1 for r in records if r["ground_truth"] == c)
        tp = cm[c][c]
        pred_n = sum(cm[g][c] for g in CLASSES)
        rec = tp / gt_total if gt_total else None
        prec = tp / pred_n if pred_n else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else 0.0
        per_class[c] = {"n": gt_total, "recall": rec, "precision": prec, "f1": f1}
        if rec is not None:
            recalls.append(rec)
        f1s.append(f1)
    out["per_class"] = per_class
    out["balanced_acc"] = sum(recalls) / len(recalls) if recalls else None
    out["macro_f1"] = sum(f1s) / len(f1s)
    # safety-oriented rates
    act = [r for r in records if r["ground_truth"] in ACTIONABLE]
    safe = [r for r in records if r["ground_truth"] == "safe"]
    out["actionable_acc"] = (sum(1 for r in act if r["decision"] == r["ground_truth"]) / len(act)) if act else None
    out["missed_actionable_rate"] = (sum(1 for r in act if r["decision"] == "safe") / len(act)) if act else None
    out["false_intervention_rate"] = (sum(1 for r in safe if r["decision"] in ACTIONABLE) / len(safe)) if safe else None
    pred_int = [r for r in records if r["decision"] in ACTIONABLE]
    out["intervention_precision"] = (sum(1 for r in pred_int if r["ground_truth"] in ACTIONABLE) / len(pred_int)) if pred_int else None
    return out


def always_x(gts, cls):
    recs = [{"ground_truth": g, "decision": cls} for g in gts]
    return metrics_from_records(recs)


def holm(pvals):
    """Holm step-down adjustment; returns adjusted p-values in input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [None] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        val = max(val, prev)
        prev = val
        adj[i] = val
    return adj


# ── peer-vote baseline ────────────────────────────────────────────

def peer_vote_action(peers, tampered):
    """Deterministic vote mirroring the prompt's decision guidance."""
    if tampered:
        stages = [0] * len(peers)
        escal = [False] * len(peers)
    else:
        stages = [int(p["outcome"]["failure_stage_max"]) for p in peers]
        escal = [bool(p["outcome"]["did_escalate"]) for p in peers]
    cnt = Counter(stages)
    top = max(cnt.values())
    stage_hat = max(s for s, c in cnt.items() if c == top)  # tie -> higher (conservative)
    p300_hat = 1 if sum(escal) >= 3 else 0
    if stage_hat >= 2:
        return "urgent"
    if stage_hat == 1 and p300_hat == 1:
        return "replace"
    return "safe"


def main():
    raw = json.loads((RESULTS / "v14_raw_results.json").read_text(encoding="utf-8"))
    pois = json.loads((RESULTS / "v14_poisoning_results.json").read_text(encoding="utf-8"))
    scen = json.loads((DATA / "v14_scenarios.json").read_text(encoding="utf-8"))
    scen = scen if isinstance(scen, list) else scen.get("scenarios", [])
    by_id = {s["scenario_id"]: s for s in scen}

    report = {}

    # 1) per-condition metrics, nominal (raw) and per rate (poisoning file)
    nominal = {}
    for c in CONDS:
        nominal[c] = metrics_from_records([r for r in raw if r["condition"] == c])
    report["nominal"] = nominal

    rates = sorted({r["rate"] for r in pois})
    per_rate = {}
    for rate in rates:
        per_rate[str(rate)] = {
            c: metrics_from_records([r for r in pois if r["condition"] == c and r["rate"] == rate])
            for c in CONDS
        }
    report["per_rate"] = per_rate

    # 2) trivial baselines (label distribution is attack-invariant)
    gts = [r["ground_truth"] for r in raw if r["condition"] == "C0"]
    report["trivial"] = {f"always_{c}": always_x(gts, c) for c in CLASSES}

    # 3) deterministic peer-vote baseline
    #    poisoned/attack flags per scenario x rate reused from the C2 rows
    flags = {}
    for r in pois:
        if r["condition"] == "C2":
            flags[(r["scenario_id"], r["rate"])] = (r.get("poisoned", False), r.get("attack_type"))
    pv = {}
    for rate in rates:
        rows_unv, rows_chk = [], []
        for s in scen:
            sid = s["scenario_id"]
            gt = s["ground_truth"]
            poisoned, atype = flags.get((sid, rate), (False, None))
            # unverified analogue: tampered peers whenever poisoned
            rows_unv.append({"ground_truth": gt,
                             "decision": peer_vote_action(s["asset_records"], tampered=poisoned)})
            # handle-checked analogue: fabricated handles detected -> restoration
            tampered_chk = poisoned and atype != "fake_hash"
            rows_chk.append({"ground_truth": gt,
                             "decision": peer_vote_action(s["asset_records"], tampered=tampered_chk)})
        pv[str(rate)] = {"peer_vote_unverified": metrics_from_records(rows_unv),
                         "peer_vote_checked": metrics_from_records(rows_chk)}
    report["peer_vote"] = pv

    # 4) Holm adjustment for the six Table VI tests
    table6 = {"0.00 C0": 0.272, "0.15 C0": 0.212, "0.30 C0": 0.027,
              "0.50 C0": 0.0023, "0.50 C1": 0.027, "0.50 C2": 0.180}
    adj = holm(list(table6.values()))
    report["holm_table6"] = {k: {"raw": v, "holm": round(a, 4)}
                             for (k, v), a in zip(table6.items(), adj)}

    # 5) v14-task latency (nominal raw)
    lat = {}
    for c in CONDS:
        rs = [r for r in raw if r["condition"] == c and r.get("latency_ms")]
        if not rs:
            lat[c] = None
            continue
        tot = sorted(r["latency_ms"] for r in rs)
        inf = sorted(r.get("latency_inference_ms", 0) for r in rs)
        ret = sorted(r.get("latency_retrieval_ms", 0) for r in rs)
        med = lambda xs: xs[len(xs)//2]
        lat[c] = {"n": len(rs), "total_ms_median": round(med(tot), 1),
                  "inference_ms_median": round(med(inf), 1),
                  "retrieval_ms_median": round(med(ret), 1),
                  "total_ms_mean": round(sum(tot)/len(tot), 1)}
    report["v14_latency"] = lat

    # 6) invalid-output (schema-noncompliant) counts
    report["invalid_outputs"] = {
        "nominal": {c: sum(1 for r in raw if r["condition"] == c and r.get("classification") == "error") for c in CONDS},
        "poisoning": {c: sum(1 for r in pois if r["condition"] == c and r.get("classification") == "error") for c in CONDS},
    }

    # 7) legacy k-ablation (for the k=5 justification)
    kfile = RESULTS / "ablation_k_summary.json"
    if kfile.exists():
        report["legacy_k_ablation"] = json.loads(kfile.read_text(encoding="utf-8"))

    # 8) duration/severity correlation across all peers (shortcut-variable check)
    durs, sevs = [], []
    for s in scen:
        for p in s["asset_records"]:
            durs.append(float(p["outcome"]["duration_s"]))
            sevs.append(int(p["outcome"]["failure_stage_max"]))
    n = len(durs)
    mx, my = sum(durs)/n, sum(sevs)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(durs, sevs)) / n
    sx = (sum((x-mx)**2 for x in durs)/n) ** 0.5
    sy = (sum((y-my)**2 for y in sevs)/n) ** 0.5
    report["peer_duration_severity_pearson_r"] = round(cov/(sx*sy), 3)

    out = RESULTS / "review_metrics.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Saved -> {out}\n")

    # ── printed summary ──────────────────────────────────────────
    def fmt(m):
        return (f"glob={m['global_acc']:.3f} bal={m['balanced_acc']:.3f} "
                f"mF1={m['macro_f1']:.3f} act={m['actionable_acc']:.3f} "
                f"missAct={m['missed_actionable_rate']:.3f} "
                f"falseInt={m['false_intervention_rate']:.3f} "
                f"intPrec={(m['intervention_precision'] if m['intervention_precision'] is not None else float('nan')):.3f}")

    print("NOMINAL:")
    for c in CONDS:
        print(f"  {c}: {fmt(nominal[c])}  invalid={nominal[c]['invalid_outputs']}")
    for k, v in report["trivial"].items():
        print(f"  {k}: {fmt(v)}")
    print(f"  peer-vote (clean): {fmt(pv['0.0']['peer_vote_unverified'])}")
    print("\nP=0.50:")
    for c in CONDS:
        print(f"  {c}: {fmt(per_rate['0.5'][c])}")
    print(f"  peer-vote unverified: {fmt(pv['0.5']['peer_vote_unverified'])}")
    print(f"  peer-vote checked:    {fmt(pv['0.5']['peer_vote_checked'])}")
    print("\nPeer-vote actionable acc by rate:")
    for rate in rates:
        u = pv[str(rate)]["peer_vote_unverified"]["actionable_acc"]
        k2 = pv[str(rate)]["peer_vote_checked"]["actionable_acc"]
        print(f"  rate={rate}: unverified={u:.3f} checked={k2:.3f}")
    print("\nHolm-adjusted Table VI:")
    for k, v in report["holm_table6"].items():
        print(f"  {k}: raw={v['raw']} holm={v['holm']}")
    print("\nv14 latency (median ms):")
    for c, v in lat.items():
        print(f"  {c}: {v}")
    print("\nC3 nominal confusion:", json.dumps(nominal["C3"]["confusion"]))
    print("C3 p=0.50 confusion:", json.dumps(per_rate["0.5"]["C3"]["confusion"]))
    print("C0 p=0.50 confusion:", json.dumps(per_rate["0.5"]["C0"]["confusion"]))
    print("\npeer duration-severity r:", report["peer_duration_severity_pearson_r"])
    if "legacy_k_ablation" in report:
        print("legacy k ablation:", json.dumps(report["legacy_k_ablation"])[:400])


if __name__ == "__main__":
    main()
