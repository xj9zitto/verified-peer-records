"""
plot_poisoning_curve.py

Plot accuracy vs adversarial-input-corruption rate for all four conditions
(C0/C1/C2/C3).  Output: results/figures/fig8_poisoning_curve.pdf.

Two panels:
  Left  — overall accuracy across all scenarios (mix of clean + poisoned)
  Right — accuracy on poisoned scenarios only (attack effectiveness)

Usage:
    python scripts/plot_poisoning_curve.py
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDS  = ["C0", "C1", "C2", "C3"]
COLORS = {"C0": "#999999", "C1": "#1f77b4", "C2": "#ff7f0e", "C3": "#2ca02c"}
MARK   = {"C0": "s",        "C1": "o",        "C2": "^",        "C3": "D"}

_GT_MAP = {
    "r1": "reroute", "r2": "reroute", "r3": "reroute", "r4": "reroute",
    "reroute": "reroute",
    "no_safe_route_available": "no_safe_route", "no_safe_route": "no_safe_route",
    "reorder": "reorder", "reorder_degradation_flag": "reorder",
    "replenish": "reorder",
    "k1": "carrier", "k2": "carrier", "k3": "carrier",
    "use k1": "carrier", "use k2": "carrier", "use k3": "carrier",
    "carrier": "carrier", "hold": "hold",
    "uncertain": "uncertain", "error": "error",
}

def _norm(s):
    return _GT_MAP.get(str(s).lower().strip(), str(s).lower().strip())


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score CI for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0, centre - half), min(1, centre + half))


def compute_curve(records, rate, cond, poisoned_only=False):
    rs = [r for r in records
          if r["rate"] == rate and r["condition"] == cond
          and r.get("classification") != "error"]
    if poisoned_only:
        rs = [r for r in rs if r.get("poisoned")]
    if not rs:
        return None, (None, None), 0
    correct = sum(1 for r in rs if _norm(r["ground_truth"]) == _norm(r["decision"]))
    n   = len(rs)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    return acc, (lo, hi), n


def make_panel(ax, records, rates, poisoned_only, title):
    for cond in CONDS:
        means, lows, highs = [], [], []
        for r in rates:
            acc, (lo, hi), n = compute_curve(records, r, cond, poisoned_only)
            means.append(acc if acc is not None else np.nan)
            lows.append(lo  if lo  is not None else np.nan)
            highs.append(hi if hi is not None else np.nan)
        means = np.array(means, dtype=float)
        lows  = np.array(lows,  dtype=float)
        highs = np.array(highs, dtype=float)
        ax.plot(rates, means, marker=MARK[cond], color=COLORS[cond],
                label=cond, lw=1.8, ms=6)
        ax.fill_between(rates, lows, highs, alpha=0.15, color=COLORS[cond])

    ax.set_xlabel("Adversarial corruption rate p")
    ax.set_ylabel("Decision accuracy")
    ax.set_xlim(min(rates) - 0.02, max(rates) + 0.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)


def main():
    results_path = RESULTS_DIR / "input_poisoning_results.json"
    if not results_path.exists():
        print(f"Missing {results_path}. Run the experiment first.")
        return
    records = json.loads(results_path.read_text())

    rates = sorted({r["rate"] for r in records})
    print(f"Loaded {len(records)} records across rates: {rates}")

    # Stacked vertically at column width for the IEEE two-column
    # layout.  Panel (b) is restricted to rates with poisoned
    # scenarios: the poisoned-only metric is undefined at p=0.
    rates_poisoned = [r for r in rates if r > 0]
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 6.4))
    make_panel(axes[0], records, rates, poisoned_only=False,
               title="(a) All scenarios")
    make_panel(axes[1], records, rates_poisoned, poisoned_only=True,
               title="(b) Poisoned scenarios only")
    plt.tight_layout()

    out_pdf = FIG_DIR / "fig8_poisoning_curve.pdf"
    out_png = FIG_DIR / "fig8_poisoning_curve.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"  Saved -> {out_pdf}")
    print(f"  Saved -> {out_png}")


if __name__ == "__main__":
    main()
