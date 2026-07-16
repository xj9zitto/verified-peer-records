"""
plot_v14_poisoning.py

Two-panel figure for the v14 input-poisoning experiment.

Left  — Overall accuracy vs corruption rate (diluted by GT=safe).
Right — Actionable accuracy: poisoned scenarios with GT in {replace, urgent}.
        This is the attack-relevant metric — where a wrong decision costs
        an unnoticed thermal runaway.  C3's verification defense dominates
        here while C0 collapses to 0.

Output: results/figures/fig9_v14_poisoning.{pdf,png}
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDS  = ["C0", "C1", "C2", "C3", "C4"]
COLORS = {"C0": "#999999", "C1": "#1f77b4", "C2": "#ff7f0e",
          "C3": "#2ca02c", "C4": "#9467bd"}
MARK   = {"C0": "s", "C1": "o", "C2": "^", "C3": "D", "C4": "P"}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0, centre - half), min(1, centre + half))


def acc(records, predicate):
    rs = [r for r in records if predicate(r)
          and r.get("classification") != "error"]
    if not rs:
        return None, (None, None), 0
    cor = sum(1 for r in rs if r["decision"] == r["ground_truth"])
    n = len(rs)
    a = cor / n
    lo, hi = wilson_ci(cor, n)
    return a, (lo, hi), n


def panel(ax, recs, rates, predicate, title, show_legend=True,
          show_ylabel=True):
    for cond in CONDS:
        means, los, his = [], [], []
        for r in rates:
            a, (lo, hi), n = acc(recs, lambda x, _c=cond, _r=r:
                                  x["rate"] == _r and x["condition"] == _c
                                  and predicate(x))
            means.append(a  if a  is not None else np.nan)
            los.append(lo   if lo is not None else np.nan)
            his.append(hi   if hi is not None else np.nan)
        ax.plot(rates, means, marker=MARK[cond], color=COLORS[cond],
                lw=1.8, ms=5.5, label=cond)
        ax.fill_between(rates, los, his, alpha=0.15, color=COLORS[cond])
    ax.set_xlabel("Adversarial corruption rate p", fontsize=9.5)
    if show_ylabel:
        ax.set_ylabel("Action accuracy", fontsize=9.5)
    ax.set_xlim(min(rates) - 0.02, max(rates) + 0.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(rates)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)


def panel_attack_type(ax, recs, rates, gt_filter):
    types = ["fake_hash", "forged_hash"]
    cols  = {"fake_hash": "#2ca02c", "forged_hash": "#d62728"}
    marks = {"fake_hash": "D",        "forged_hash": "X"}
    for atype in types:
        means, los, his = [], [], []
        # rate=0 has no attacks, skip
        rates_atype = [r for r in rates if r > 0]
        for r in rates_atype:
            a, (lo, hi), n = acc(recs, lambda x, _a=atype, _r=r:
                                  x["rate"] == _r and x["condition"] == "C3"
                                  and x.get("poisoned")
                                  and x.get("attack_type") == _a
                                  and gt_filter(x))
            means.append(a  if a  is not None else np.nan)
            los.append(lo   if lo is not None else np.nan)
            his.append(hi   if hi is not None else np.nan)
        ax.plot(rates_atype, means, marker=marks[atype], color=cols[atype],
                lw=1.8, ms=6,
                label=("fabricated-provenance" if atype == "fake_hash"
                       else "valid-prov. substitution"))
        ax.fill_between(rates_atype, los, his, alpha=0.15, color=cols[atype])
    ax.set_xlabel("Adversarial corruption rate p", fontsize=9.5)
    ax.set_xlim(0.10, max(rates) + 0.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([r for r in rates if r > 0])
    ax.set_title("(c) C3 by attack type", fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def main():
    recs = json.loads((RESULTS_DIR / "v14_poisoning_results.json").read_text())
    c4_path = RESULTS_DIR / "v14_poisoning_c4_results.json"
    if c4_path.exists():
        recs += json.loads(c4_path.read_text())
    rates = sorted({r["rate"] for r in recs})
    print(f"Loaded {len(recs)} records, rates {rates}")

    # Authored near final rendered width (0.85\textwidth) so fonts
    # print at ~8 pt in the IEEE two-column layout.
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.7), sharey=True)
    panel(axes[0], recs, rates, lambda r: True,
          "(a) All scenarios")
    panel(axes[1], recs, rates,
          lambda r: r["ground_truth"] in ("replace", "urgent"),
          "(b) Actionable subset", show_legend=False, show_ylabel=False)
    panel_attack_type(axes[2], recs, rates,
                      lambda r: r["ground_truth"] in ("replace", "urgent"))
    plt.tight_layout()
    pdf = FIG_DIR / "fig9_poisoning.pdf"
    png = FIG_DIR / "fig9_poisoning.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    print(f"  Saved -> {pdf}")
    print(f"  Saved -> {png}")


if __name__ == "__main__":
    main()
