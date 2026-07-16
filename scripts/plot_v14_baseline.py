"""
plot_v14_baseline.py

Per-class action-accuracy bars for the v14 hybrid (predict+decide) task
in the benign condition (no adversarial corruption).

Output: results/figures/fig10_v14_baseline_per_class.{pdf,png}

This figure visually supports the key qualitative observation in
Section 6 of the manuscript: C0 dominates on the "safe" class but
fails on the actionable classes (replace, urgent), while LLM-based
conditions are stronger on replace and urgent at the cost of
over-escalating safe.
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
CLASSES = ["safe", "replace", "urgent"]
COLORS = {"C0": "#999999", "C1": "#1f77b4", "C2": "#ff7f0e", "C3": "#2ca02c"}
HATCHES = {"C0": "//", "C1": "..", "C2": "--", "C3": "\\\\"}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0, centre - half), min(1, centre + half))


def per_class_acc(records, cond, gt_class):
    rs = [r for r in records
          if r["condition"] == cond
          and r["ground_truth"] == gt_class
          and r.get("classification") != "error"]
    n = len(rs)
    if n == 0:
        return None, (None, None), 0
    cor = sum(1 for r in rs if r["decision"] == r["ground_truth"])
    return cor/n, wilson_ci(cor, n), n


def main():
    raw_path = RESULTS_DIR / "v14_raw_results.json"
    if not raw_path.exists():
        print(f"Missing {raw_path}.  Run the v14 baseline experiment first.")
        return
    recs = json.loads(raw_path.read_text())
    print(f"Loaded {len(recs)} records")

    # Authored near column width so fonts print at ~8 pt in the
    # IEEE two-column layout.
    fig, ax = plt.subplots(figsize=(4.4, 3.5))

    n_cls = len(CLASSES)
    n_cnd = len(CONDS)
    bar_w = 0.18
    group_centers = np.arange(n_cls)

    for i, cond in enumerate(CONDS):
        means, los, his, ns = [], [], [], []
        for cls in CLASSES:
            acc, (lo, hi), n = per_class_acc(recs, cond, cls)
            means.append(acc  if acc is not None else 0)
            los.append(lo   if lo  is not None else 0)
            his.append(hi   if hi  is not None else 0)
            ns.append(n)

        positions = group_centers + (i - (n_cnd - 1) / 2.0) * bar_w
        means = np.array(means); los = np.array(los); his = np.array(his)

        bars = ax.bar(positions, means, width=bar_w,
                       color=COLORS[cond], hatch=HATCHES[cond],
                       edgecolor="black", linewidth=0.6,
                       label=cond, alpha=0.85)
        # error bars (asymmetric)
        ax.errorbar(positions, means,
                    yerr=[means - los, his - means],
                    fmt="none", ecolor="black", capsize=2, linewidth=0.8)
        # value labels above the CI upper whisker so the error bars
        # cannot strike through the text; C3 is lifted because its
        # values coincide with C2 at nominal and the labels would
        # otherwise collide on adjacent bars
        dy = 0.07 if cond == "C3" else 0.015
        for x, y, hi in zip(positions, means, his):
            ax.text(x, hi + dy, f"{y:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # x-axis class labels with n
    class_labels = []
    for cls in CLASSES:
        n_total = sum(1 for r in recs
                      if r["ground_truth"] == cls and r["condition"] == "C0"
                      and r.get("classification") != "error")
        class_labels.append(f"{cls}\n(n={n_total})")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(class_labels)
    ax.set_ylabel("Action accuracy", fontsize=10)
    ax.set_ylim(0, 1.16)
    ax.tick_params(labelsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    # legend above the axes so it cannot occlude bars or value labels
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              fontsize=9, framealpha=1.0, borderaxespad=0.2,
              handletextpad=0.5, columnspacing=1.0)

    # Annotate the central finding: actionable class collapse pattern
    ax.axvline(x=group_centers[0] + bar_w * 2, ymin=0, ymax=1,
               color="gray", linestyle=":", alpha=0.5, lw=0.9)
    ax.text(group_centers[0] + bar_w * 2 + 0.05, 1.145,
            "← safe class | actionable classes →",
            ha="left", va="top", fontsize=8.5, color="dimgray",
            style="italic")

    plt.tight_layout()
    pdf = FIG_DIR / "fig10_baseline_per_class.pdf"
    png = FIG_DIR / "fig10_baseline_per_class.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    print(f"  Saved -> {pdf}")
    print(f"  Saved -> {png}")


if __name__ == "__main__":
    main()
