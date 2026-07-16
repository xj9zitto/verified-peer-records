"""
analyze_c4.py

Analysis of the C4 (content-bound verification + chain-based recovery)
condition against the existing C0-C3 poisoning results.

Reports, per corruption rate:
  - overall and actionable-subset accuracy for all five conditions
  - C4 attack-type decomposition (fabricated vs valid-substitution)
  - paired exact McNemar tests of C4 vs each baseline on the
    actionable subset (the primary comparison is C4 vs C2)
  - detection/recovery bookkeeping (tamper alerts, peers recovered)

Output: results/c4_analysis.json + printed report.
"""

import json
import math
from pathlib import Path

RESULTS = Path("results")
BASE_FILE = RESULTS / "v14_poisoning_results.json"
C4_FILE   = RESULTS / "v14_poisoning_c4_results.json"

ACTIONABLE = ("replace", "urgent")


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    return min(1.0, 2 * tail)


def acc(rows):
    ok = [r for r in rows if r.get("classification") != "error"]
    if not ok:
        return None, 0
    return sum(1 for r in ok if r["decision"] == r["ground_truth"]) / len(ok), len(ok)


def main():
    base = json.loads(BASE_FILE.read_text(encoding="utf-8"))
    c4 = json.loads(C4_FILE.read_text(encoding="utf-8"))
    recs = base + c4
    conds = ["C0", "C1", "C2", "C3", "C4"]
    rates = sorted({r["rate"] for r in recs})

    out = {"by_rate": {}, "actionable": {}, "attack_decomp": {},
           "mcnemar_actionable": {}, "recovery": {}}

    print("Overall accuracy by rate:")
    print("rate   " + "  ".join(f"{c:>7}" for c in conds))
    for rate in rates:
        row = {}
        for c in conds:
            a, n = acc([r for r in recs if r["condition"] == c and r["rate"] == rate])
            row[c] = round(a, 4) if a is not None else None
        out["by_rate"][str(rate)] = row
        print(f"{rate:<5}  " + "  ".join(f"{row[c]:>7.3f}" if row[c] is not None else "     NA" for c in conds))

    print("\nActionable-subset accuracy by rate:")
    print("rate   " + "  ".join(f"{c:>7}" for c in conds))
    for rate in rates:
        row = {}
        for c in conds:
            a, n = acc([r for r in recs if r["condition"] == c and r["rate"] == rate
                        and r["ground_truth"] in ACTIONABLE])
            row[c] = round(a, 4) if a is not None else None
        out["actionable"][str(rate)] = row
        print(f"{rate:<5}  " + "  ".join(f"{row[c]:>7.3f}" if row[c] is not None else "     NA" for c in conds))

    print("\nC4 attack decomposition (poisoned actionable):")
    for rate in [r for r in rates if r > 0]:
        sub = {}
        for atype in ("fake_hash", "forged_hash"):
            rows = [r for r in c4 if r["rate"] == rate and r.get("attack_type") == atype
                    and r["ground_truth"] in ACTIONABLE]
            a, n = acc(rows)
            sub[atype] = {"acc": round(a, 4) if a is not None else None, "n": n}
        out["attack_decomp"][str(rate)] = sub
        print(f"  rate={rate}: fabricated={sub['fake_hash']}  valid-subst={sub['forged_hash']}")

    print("\nPaired McNemar on actionable subset (C4 vs comparator):")
    by_key = {}
    for r in recs:
        if r.get("classification") == "error":
            continue
        by_key[(r["condition"], r["rate"], r["scenario_id"])] = \
            r["decision"] == r["ground_truth"]
    gts = {r["scenario_id"]: r["ground_truth"] for r in c4}
    act_ids = [sid for sid, g in gts.items() if g in ACTIONABLE]
    for rate in rates:
        out["mcnemar_actionable"][str(rate)] = {}
        for comp in ("C0", "C1", "C2", "C3"):
            b = c_ = 0
            for sid in act_ids:
                k4, kc = ("C4", rate, sid), (comp, rate, sid)
                if k4 not in by_key or kc not in by_key:
                    continue
                if by_key[k4] and not by_key[kc]:
                    b += 1
                elif by_key[kc] and not by_key[k4]:
                    c_ += 1
            p = exact_mcnemar(b, c_)
            out["mcnemar_actionable"][str(rate)][comp] = \
                {"b": b, "c": c_, "p": round(p, 5)}
            print(f"  rate={rate} C4 vs {comp}: b={b:>2} c={c_:>2} p={p:.4f}")

    print("\nDetection / recovery bookkeeping (C4, poisoned rows):")
    for rate in [r for r in rates if r > 0]:
        rows = [r for r in c4 if r["rate"] == rate and r.get("poisoned")]
        alerts = sum(r.get("tamper_alerts", 0) for r in rows)
        recov = sum(r.get("peers_recovered", 0) for r in rows)
        out["recovery"][str(rate)] = {"poisoned_scenarios": len(rows),
                                      "tamper_alerts": alerts,
                                      "peers_recovered": recov}
        print(f"  rate={rate}: {len(rows)} poisoned scenarios, "
              f"{alerts} alerts, {recov} peers recovered from chain "
              f"({recov}/{alerts})")

    (RESULTS / "c4_analysis.json").write_text(json.dumps(out, indent=1),
                                              encoding="utf-8")
    print(f"\nSaved -> {RESULTS / 'c4_analysis.json'}")


if __name__ == "__main__":
    main()
