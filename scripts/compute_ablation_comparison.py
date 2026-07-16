"""
compute_ablation_comparison.py

Compare C3-with-outcomes (full verified-peer-trajectory primitive)
against C3-state-only (peer outcomes hidden from LLM).  Prints a
side-by-side table for overall, actionable, poisoned-actionable, and
attack-type slices, and writes a structured summary to
results/v14_ablation_comparison.json.
"""
import json
from pathlib import Path

RESULTS = Path("results")

with_out = json.loads((RESULTS / "v14_poisoning_results.json").read_text())
no_out   = json.loads((RESULTS / "v14_poisoning_ablation_results.json").read_text())

# Restrict the with-outcomes set to C3 only for fair comparison
with_out = [r for r in with_out if r["condition"] == "C3"]
no_out   = [r for r in no_out if r["condition"] == "C3"]

print(f"C3-with-outcomes records: {len(with_out)}")
print(f"C3-state-only records:    {len(no_out)}")


def correct(r):
    return r["decision"] == r["ground_truth"]


def slice_acc(records, rate, predicate):
    rs = [r for r in records
          if r["rate"] == rate and predicate(r)
          and r.get("classification") != "error"]
    if not rs:
        return None, 0
    c = sum(1 for r in rs if correct(r))
    return c / len(rs), len(rs)


print("\n" + "=" * 78)
print("OVERALL ACCURACY (all scenarios)")
print("=" * 78)
print(f"  {'rate':<6}  {'with_outcomes':>18}  {'state_only':>18}  {'delta':>10}")
for rate in [0.0, 0.15, 0.30, 0.50]:
    a, na = slice_acc(with_out, rate, lambda r: True)
    b, nb = slice_acc(no_out,   rate, lambda r: True)
    d = (b - a) if a is not None and b is not None else None
    print(f"  {rate:<6.2f}  {a:>10.3f} (n={na:>3})  "
          f"{b:>10.3f} (n={nb:>3})  {d:>+10.3f}")

print("\n" + "=" * 78)
print("ACTIONABLE SUBSET (GT in {replace, urgent})")
print("=" * 78)
print(f"  {'rate':<6}  {'with_outcomes':>18}  {'state_only':>18}  {'delta':>10}")
for rate in [0.0, 0.15, 0.30, 0.50]:
    a, na = slice_acc(with_out, rate,
                      lambda r: r["ground_truth"] in ("replace", "urgent"))
    b, nb = slice_acc(no_out,   rate,
                      lambda r: r["ground_truth"] in ("replace", "urgent"))
    d = (b - a) if a is not None and b is not None else None
    print(f"  {rate:<6.2f}  {a:>10.3f} (n={na:>3})  "
          f"{b:>10.3f} (n={nb:>3})  {d:>+10.3f}")

print("\n" + "=" * 78)
print("POISONED + ACTIONABLE (where attack actually has bite)")
print("=" * 78)
print(f"  {'rate':<6}  {'with_outcomes':>18}  {'state_only':>18}  {'delta':>10}")
for rate in [0.15, 0.30, 0.50]:
    a, na = slice_acc(with_out, rate,
                      lambda r: r.get("poisoned")
                      and r["ground_truth"] in ("replace", "urgent"))
    b, nb = slice_acc(no_out,   rate,
                      lambda r: r.get("poisoned")
                      and r["ground_truth"] in ("replace", "urgent"))
    d = (b - a) if a is not None and b is not None else None
    print(f"  {rate:<6.2f}  {a:>10.3f} (n={na:>3})  "
          f"{b:>10.3f} (n={nb:>3})  {d:>+10.3f}")

print("\n" + "=" * 78)
print("ATTACK-TYPE DECOMPOSITION ON ACTIONABLE+POISONED")
print("=" * 78)
for rate in [0.15, 0.30, 0.50]:
    print(f"\n  rate = {rate}")
    print(f"  {'attack':<14}  {'with_outcomes':>18}  {'state_only':>18}  {'delta':>10}")
    for atype in ("fake_hash", "forged_hash"):
        a, na = slice_acc(
            with_out, rate,
            lambda r, _a=atype: r.get("poisoned")
            and r["ground_truth"] in ("replace", "urgent")
            and r.get("attack_type") == _a,
        )
        b, nb = slice_acc(
            no_out, rate,
            lambda r, _a=atype: r.get("poisoned")
            and r["ground_truth"] in ("replace", "urgent")
            and r.get("attack_type") == _a,
        )
        d = (b - a) if a is not None and b is not None else None
        a_str = f"{a:>10.3f} (n={na:>2})" if a is not None else f"{'NA':>18}"
        b_str = f"{b:>10.3f} (n={nb:>2})" if b is not None else f"{'NA':>18}"
        d_str = f"{d:>+10.3f}" if d is not None else f"{'NA':>10}"
        print(f"  {atype:<14}  {a_str}  {b_str}  {d_str}")

print("\n" + "=" * 78)
print("PER-GT-CLASS BREAKDOWN AT RATE 0.50 (POISONED ONLY)")
print("=" * 78)
print(f"  {'gt':<10}  {'with_outcomes':>18}  {'state_only':>18}  {'delta':>10}")
per_class = {}
for gt in ["safe", "replace", "urgent"]:
    a, na = slice_acc(
        with_out, 0.50,
        lambda r, _g=gt: r.get("poisoned") and r["ground_truth"] == _g,
    )
    b, nb = slice_acc(
        no_out, 0.50,
        lambda r, _g=gt: r.get("poisoned") and r["ground_truth"] == _g,
    )
    d = (b - a) if a is not None and b is not None else None
    a_str = f"{a:>10.3f} (n={na:>3})" if a is not None else f"{'NA':>18}"
    b_str = f"{b:>10.3f} (n={nb:>3})" if b is not None else f"{'NA':>18}"
    d_str = f"{d:>+10.3f}" if d is not None else f"{'NA':>10}"
    print(f"  {gt:<10}  {a_str}  {b_str}  {d_str}")
    per_class[gt] = {"with_outcomes": a, "state_only": b,
                     "delta": d, "n": na}

# ── Save structured summary ──────────────────────────────────
summary = {"overall": {}, "actionable": {}, "poisoned_actionable": {},
           "attack_type_at_p50": {}, "per_class_at_p50_poisoned": per_class}
for rate in [0.0, 0.15, 0.30, 0.50]:
    a, _ = slice_acc(with_out, rate, lambda r: True)
    b, _ = slice_acc(no_out,   rate, lambda r: True)
    summary["overall"][f"{rate:.2f}"] = {"with_outcomes": a,
                                          "state_only": b,
                                          "delta": (b - a) if a is not None and b is not None else None}
    a, _ = slice_acc(with_out, rate, lambda r: r["ground_truth"] in ("replace", "urgent"))
    b, _ = slice_acc(no_out,   rate, lambda r: r["ground_truth"] in ("replace", "urgent"))
    summary["actionable"][f"{rate:.2f}"] = {"with_outcomes": a,
                                            "state_only": b,
                                            "delta": (b - a) if a is not None and b is not None else None}
for rate in [0.15, 0.30, 0.50]:
    a, _ = slice_acc(with_out, rate, lambda r: r.get("poisoned") and r["ground_truth"] in ("replace", "urgent"))
    b, _ = slice_acc(no_out,   rate, lambda r: r.get("poisoned") and r["ground_truth"] in ("replace", "urgent"))
    summary["poisoned_actionable"][f"{rate:.2f}"] = {"with_outcomes": a,
                                                      "state_only": b,
                                                      "delta": (b - a) if a is not None and b is not None else None}

for atype in ("fake_hash", "forged_hash"):
    a, na = slice_acc(with_out, 0.50,
                      lambda r, _a=atype: r.get("poisoned")
                      and r["ground_truth"] in ("replace", "urgent")
                      and r.get("attack_type") == _a)
    b, nb = slice_acc(no_out, 0.50,
                      lambda r, _a=atype: r.get("poisoned")
                      and r["ground_truth"] in ("replace", "urgent")
                      and r.get("attack_type") == _a)
    summary["attack_type_at_p50"][atype] = {"with_outcomes": a,
                                             "state_only": b,
                                             "delta": (b - a) if a is not None and b is not None else None,
                                             "n": na}

out_path = RESULTS / "v14_ablation_comparison.json"
out_path.write_text(json.dumps(summary, indent=2))
print(f"\nSaved -> {out_path}")
