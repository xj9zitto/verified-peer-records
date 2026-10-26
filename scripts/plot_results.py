"""
plot_results.py
Generate IEEE TII figures from experiment results.

Figures produced (saved to results/figures/):
  fig1_accuracy_bars.pdf    — main accuracy comparison C0–C3 with 95% CI
  fig2_latency_stack.pdf    — latency breakdown (retrieval / inference / overhead)
  fig3_type_heatmap.pdf     — per-type accuracy heatmap (4 types × 4 conditions)
  fig4_k_ablation.pdf       — k-ablation: accuracy + τ vs k
  fig5_prompt_stability.pdf — prompt variation: accuracy + τ for v1/v2/v3

Usage:
    python scripts/plot_results.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── IEEE TII style ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         7,
    "axes.titlesize":    7.5,
    "axes.labelsize":    7,
    "xtick.labelsize":   6.5,
    "ytick.labelsize":   6.5,
    "legend.fontsize":   8,
    "lines.linewidth":   1.2,
    "lines.markersize":  4,
    "axes.linewidth":    0.7,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.4,
    "pdf.fonttype":      42,   # embeds fonts
    "ps.fonttype":       42,
})

# Column widths for IEEE double-column format
COL1 = 3.5    # single column (inches)
COL2 = 7.16   # double column (inches)

# Colour palette — greyscale-safe, print-friendly
C_COLORS = {
    "C0": "#4e4e4e",   # dark grey
    "C1": "#1f77b4",   # blue
    "C2": "#ff7f0e",   # orange
    "C3": "#2ca02c",   # green
}
HATCH = {
    "C0": "////",
    "C1": "....",
    "C2": "----",
    "C3": "",
}

RESULTS = Path("results")
OUT     = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Ground-truth normaliser ────────────────────────────────────
_GT_MAP = {
    "r1": "reroute", "r2": "reroute", "r3": "reroute", "r4": "reroute",
    "reroute": "reroute",
    "no_safe_route_available": "no_safe_route",
    "no_safe_route":           "no_safe_route",
    "reorder":                 "reorder",
    "reorder_degradation_flag":"reorder",
    "replenish":               "reorder",
    "k1": "carrier", "k2": "carrier", "k3": "carrier",
    "use k1": "carrier", "use k2": "carrier", "use k3": "carrier",
    "carrier": "carrier",
    "hold":    "hold",
    "uncertain":"uncertain",
    "error":   "error",
}

def _norm(label: str) -> str:
    return _GT_MAP.get(str(label).lower().strip(), str(label).lower().strip())


def _dedup(records: list, key: str = "scenario_id") -> list:
    """Keep last record per key (handles double-run artefacts)."""
    seen: dict = {}
    for r in records:
        seen[r[key]] = r
    return list(seen.values())


# ── Load data ──────────────────────────────────────────────────

def load_main() -> dict:
    with open(RESULTS / "metrics.json") as f:
        return json.load(f)

def load_raw() -> list:
    with open(RESULTS / "raw_results.json") as f:
        return json.load(f)

def load_ablation_k() -> dict:
    """Returns {k_total: [records]} — deduped, errors excluded."""
    k_configs = {3: (2,1), 5: (3,2), 8: (5,3), 12: (8,4), 16: (10,6)}
    result = {}

    # k=8 from baseline
    raw = load_raw()
    result[8] = [r for r in raw
                 if r["condition"] == "C3" and r["classification"] != "error"]

    for k in [3, 5, 12, 16]:
        path = RESULTS / f"ablation_k{k}.json"
        if not path.exists():
            continue
        with open(path) as f:
            recs = json.load(f)
        recs = _dedup(recs)
        result[k] = [r for r in recs if r.get("classification") != "error"]

    return result

def load_ablation_prompt() -> dict:
    """Returns {variant: [records]} — deduped, errors excluded."""
    raw = load_raw()
    result = {
        "v1": [r for r in raw
               if r["condition"] == "C3" and r["classification"] != "error"]
    }
    for v in ["v2", "v3"]:
        path = RESULTS / f"ablation_prompt_{v}.json"
        if not path.exists():
            continue
        with open(path) as f:
            recs = json.load(f)
        recs = _dedup(recs)
        result[v] = [r for r in recs if r.get("classification") != "error"]
    return result


def _acc_ci(records):
    correct = sum(1 for r in records
                  if _norm(r["ground_truth"]) == _norm(r["decision"]))
    n   = len(records)
    acc = correct / n if n else 0
    # Wilson score 95% CI
    z = 1.96
    denom = 1 + z**2 / n
    center = (acc + z**2 / (2*n)) / denom
    half   = z * np.sqrt(acc*(1-acc)/n + z**2/(4*n**2)) / denom
    return acc, max(0, center - half), min(1, center + half)


# ── Figure 1: Main accuracy bars (coarse + strict) ────────────

def _draw_sig_bracket(ax, x1, x2, y, label, color="black", fontsize=5.5):
    """Draw a significance bracket between bar centers x1 and x2 at height y."""
    tick_h = 0.015
    ax.plot([x1, x1, x2, x2], [y, y + tick_h, y + tick_h, y],
            lw=0.8, color=color)
    ax.text((x1 + x2) / 2, y + tick_h + 0.005, label,
            ha="center", va="bottom", fontsize=fontsize, color=color)


def fig1_accuracy(metrics: dict, sig: dict | None = None) -> None:
    conditions = ["C0", "C1", "C2", "C3"]
    labels     = ["C0\n(Rule)", "C1\n(LLM)", "C2\n(RAG)", "C3\n(PureChain)"]

    # Coarse (category-level) accuracy
    coarse_accs = [metrics[c]["accuracy"]["mean"] for c in conditions]
    coarse_ci_lo = [metrics[c]["accuracy"]["mean"] - metrics[c]["accuracy"]["ci_95"][0]
                    for c in conditions]
    coarse_ci_hi = [metrics[c]["accuracy"]["ci_95"][1] - metrics[c]["accuracy"]["mean"]
                    for c in conditions]

    # Strict (specific-label) accuracy
    strict_accs = [
        metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["mean"]
        for c in conditions
    ]
    strict_ci_lo = [
        metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["mean"]
        - metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["ci_95"][0]
        for c in conditions
    ]
    strict_ci_hi = [
        metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["ci_95"][1]
        - metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["mean"]
        for c in conditions
    ]

    fig, ax = plt.subplots(figsize=(COL1 * 1.15, 2.7))
    x = np.arange(len(conditions))
    w = 0.38

    bars_c = ax.bar(
        x - w/2, coarse_accs, width=w,
        color=[C_COLORS[c] for c in conditions],
        hatch=[HATCH[c] for c in conditions],
        edgecolor="white", linewidth=0.6,
        yerr=[coarse_ci_lo, coarse_ci_hi],
        error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9,
                  "ecolor": "black", "zorder": 5},
        zorder=3, label="Category acc.",
    )
    bars_s = ax.bar(
        x + w/2, strict_accs, width=w,
        color=[C_COLORS[c] for c in conditions],
        alpha=0.45,
        edgecolor=[C_COLORS[c] for c in conditions],
        linewidth=1.0,
        yerr=[strict_ci_lo, strict_ci_hi],
        error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9,
                  "ecolor": "black", "zorder": 5},
        zorder=3, label="Specific acc.",
    )

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Decision Accuracy")
    ax.set_title("(a) Accuracy by Experimental Condition")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.5, loc="upper left", framealpha=0.85, ncol=2)

    for bar, acc in zip(bars_c, coarse_accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.04,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=5, fontweight="bold")
    for bar, acc in zip(bars_s, strict_accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.04,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=5, color="#555555")

    # Significance brackets vs C3 (index 3)
    if sig and "overall_mcnemar" in sig:
        mc = sig["overall_mcnemar"]
        alpha_bf = sig.get("alpha_bonferroni", 0.0167)
        c3_x = x[3]  # C3 bar center
        bracket_y = [0.98, 1.04, 1.10]
        for bi, cond_i in enumerate([0, 1, 2]):   # C0, C1, C2
            key = f"C3_vs_C{cond_i}"
            if key not in mc:
                continue
            p  = mc[key]["p_value"]
            st = _sig_stars(p, alpha_bf)
            _draw_sig_bracket(ax, x[cond_i], c3_x, bracket_y[bi], st,
                              color="black", fontsize=5.5)

    fig.tight_layout(pad=0.5)
    out = OUT / "fig1_accuracy_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Figure 2: Latency stacked bar ─────────────────────────────

def fig2_latency(metrics: dict) -> None:
    conditions = ["C1", "C2", "C3"]
    labels     = ["C1\n(LLM only)", "C2\n(RAG, no verify)", "C3\n(PureChain RAG)"]

    retrieval = [metrics[c]["latency"][c]["retrieval_ms"] for c in conditions]
    inference = [metrics[c]["latency"][c]["inference_ms"] for c in conditions]
    totals    = [metrics[c]["latency"][c]["total_ms"]     for c in conditions]
    overhead  = [t - r - i for t, r, i in zip(totals, retrieval, inference)]

    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    x = np.arange(len(conditions))
    w = 0.5

    b1 = ax.bar(x, retrieval, w, label="Retrieval + Verify",
                color="#4e9dc0", hatch="////", edgecolor="white", linewidth=0.6, zorder=3)
    b2 = ax.bar(x, inference, w, bottom=retrieval, label="LLM Inference",
                color="#2ca02c", hatch="", edgecolor="white", linewidth=0.6, zorder=3)
    b3 = ax.bar(x, overhead, w,
                bottom=[r+i for r,i in zip(retrieval, inference)],
                label="Overhead", color="#d6d6d6", hatch="....", edgecolor="white",
                linewidth=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-End Latency Breakdown")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    for xi, total in zip(x, totals):
        ax.text(xi, total + 30, f"{total:.0f}", ha="center", va="bottom",
                fontsize=5, fontweight="bold")

    fig.tight_layout(pad=0.5)
    out = OUT / "fig2_latency_stack.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Figure 3: Per-type accuracy heatmap ───────────────────────

def fig3_type_heatmap(metrics: dict) -> None:
    scenario_types = ["reorder", "reroute", "path", "carrier"]
    conditions     = ["C0", "C1", "C2", "C3"]
    type_labels    = ["Reorder", "Reroute", "Path\nSelect.", "Carrier\nChoice"]
    cond_labels    = ["C0\n(Rule)", "C1\n(LLM)", "C2\n(RAG)", "C3\n(Pure\nChain)"]

    # Use strict accuracy; coarse shown in parentheses
    strict_matrix = np.array([
        [metrics["by_type"][t][c].get("strict_accuracy", metrics["by_type"][t][c]["accuracy"])
         for c in conditions]
        for t in scenario_types
    ])
    coarse_matrix = np.array([
        [metrics["by_type"][t][c]["accuracy"] for c in conditions]
        for t in scenario_types
    ])

    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    im = ax.imshow(strict_matrix, aspect="auto", cmap="YlGn", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(cond_labels, fontsize=7)
    ax.set_yticks(range(len(scenario_types)))
    ax.set_yticklabels(type_labels, fontsize=7)
    ax.set_title("Strict Accuracy by Type\n(category acc. in parentheses)", fontsize=8)

    for i in range(len(scenario_types)):
        for j in range(len(conditions)):
            sv = strict_matrix[i, j]
            cv = coarse_matrix[i, j]
            color = "white" if sv > 0.6 else "black"
            # Main value: strict
            ax.text(j, i - 0.12, f"{sv:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")
            # Sub-value: coarse in lighter text
            if abs(sv - cv) > 0.01:
                ax.text(j, i + 0.28, f"({cv:.2f})", ha="center", va="center",
                        fontsize=5.5, color=color, alpha=0.75)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Strict Accuracy", fontsize=6.5)

    fig.tight_layout(pad=0.5)
    out = OUT / "fig3_type_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Figure 4: k-ablation ──────────────────────────────────────

def fig4_k_ablation(k_data: dict) -> None:
    k_vals = sorted(k_data.keys())
    accs, taus, lats = [], [], []

    for k in k_vals:
        recs = k_data[k]
        correct = sum(1 for r in recs
                      if _norm(r["ground_truth"]) == _norm(r["decision"]))
        accs.append(correct / len(recs) if recs else 0)
        taus.append(np.mean([r["trace_rate"] for r in recs]) if recs else 0)
        lats.append(np.mean([r["latency_ms"] for r in recs]) if recs else 0)

    fig, ax1 = plt.subplots(figsize=(COL1, 2.5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(k_vals, accs, "s-", color="#2ca02c", label="Accuracy", zorder=5)
    l2, = ax1.plot(k_vals, taus, "^--", color="#1f77b4", label="Trace rate τ", zorder=5)
    l3, = ax2.plot(k_vals, lats, "o:", color="#d62728", label="Latency (ms)", alpha=0.7, zorder=4)

    # Baseline marker
    baseline_idx = k_vals.index(8) if 8 in k_vals else None
    if baseline_idx is not None:
        ax1.axvline(8, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax1.text(8.2, 0.78, "baseline\nk=8", fontsize=5.5, color="grey")

    ax1.set_xlabel("Total Retrieved Records k")
    ax1.set_ylabel("Accuracy / Trace Rate τ")
    ax2.set_ylabel("Latency (ms)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("(d) k-Ablation Study (Metric 8)")
    ax1.set_ylim(0.7, 1.02)
    ax2.set_ylim(2500, 4500)
    ax1.set_xticks(k_vals)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.85, fontsize=5)

    fig.tight_layout(pad=0.5)
    out = OUT / "fig4_k_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Figure 5: Prompt variation ─────────────────────────────────

def fig5_prompt_stability(pv_data: dict) -> None:
    variants = ["v1", "v2", "v3"]
    v_labels = ["v1\n(Structured)", "v2\n(Step-by-step)", "v3\n(Concise NL)"]
    accs, taus, lats = [], [], []

    for v in variants:
        recs = pv_data.get(v, [])
        if not recs:
            accs.append(0); taus.append(0); lats.append(0)
            continue
        correct = sum(1 for r in recs
                      if _norm(r["ground_truth"]) == _norm(r["decision"]))
        accs.append(correct / len(recs))
        taus.append(np.mean([r["trace_rate"] for r in recs]))
        lats.append(np.mean([r["latency_ms"] for r in recs]))

    x  = np.arange(len(variants))
    w  = 0.3

    fig, ax1 = plt.subplots(figsize=(COL1, 2.6))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w/2, accs, w, label="Accuracy",
                 color="#2ca02c", hatch="", edgecolor="white", zorder=3)
    b2 = ax1.bar(x + w/2, taus, w, label="Trace rate τ",
                 color="#1f77b4", hatch="////", edgecolor="white", zorder=3)
    l1, = ax2.plot(x, lats, "o--", color="#d62728", label="Latency (ms)", zorder=5)

    ax1.set_ylim(0.5, 1.05)
    ax2.set_ylabel("Latency (ms)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(1500, 5500)
    ax1.set_xticks(x)
    ax1.set_xticklabels(v_labels)
    ax1.set_ylabel("Accuracy / Trace Rate τ")
    ax1.set_title("(e) Prompt Variation Stability (Metric 9)")
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)

    for bar, acc in zip(b1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    # Accuracy std annotation
    acc_std = np.std(accs)
    ax1.text(0.97, 0.07, f"Acc σ = {acc_std:.4f}", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=5.5, style="italic",
             bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="grey", alpha=0.8))

    lines  = [b1, b2, l1]
    labels = ["Accuracy", "Trace rate τ", "Latency (ms)"]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.85, fontsize=5)

    fig.tight_layout(pad=0.5)
    out = OUT / "fig5_prompt_stability.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Figure 6: Cohen's κ comparison ────────────────────────────

def fig6_cohens_kappa(metrics: dict) -> None:
    kappas = metrics["cohens_kappa"]
    pairs  = ["C0_vs_C3", "C1_vs_C3", "C2_vs_C3"]
    labels = ["C0 vs C3", "C1 vs C3", "C2 vs C3"]
    values = [kappas[p]["kappa"] for p in pairs]
    colors = [C_COLORS["C0"], C_COLORS["C1"], C_COLORS["C2"]]
    hatches = [HATCH["C0"], HATCH["C1"], HATCH["C2"]]

    fig, ax = plt.subplots(figsize=(COL1 * 0.75, 2.3))
    x = np.arange(len(pairs))
    bars = ax.bar(x, values, 0.5, color=colors, hatch=hatches,
                  edgecolor="white", linewidth=0.6, zorder=3)

    # Reference lines for κ interpretation
    for kappa_ref, label in [(0.2, "Fair"), (0.4, "Moderate"),
                               (0.6, "Substantial"), (0.8, "Almost perfect")]:
        ax.axhline(kappa_ref, color="grey", linewidth=0.6, linestyle=":", alpha=0.7)
        ax.text(2.35, kappa_ref + 0.01, label, fontsize=5.5, color="grey", va="bottom")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cohen's κ")
    ax.set_title("(f) Inter-Condition Agreement (κ vs C3)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=5, fontweight="bold")

    fig.tight_layout(pad=0.5)
    out = OUT / "fig6_cohens_kappa.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Combined 2×3 panel (for paper body) ───────────────────────

def fig_combined(metrics: dict, k_data: dict, pv_data: dict, sig: dict | None = None) -> None:
    """
    Single double-column figure with all 6 panels arranged 2 rows × 3 cols.
    Suitable for the main results section of an IEEE TII paper.
    """
    fig = plt.figure(figsize=(COL2, 5.2))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.42)

    conditions = ["C0", "C1", "C2", "C3"]

    # ── Panel (a): accuracy bars (coarse + strict) ────────────
    ax = fig.add_subplot(gs[0, 0])
    coarse_accs = [metrics[c]["accuracy"]["mean"] for c in conditions]
    strict_accs = [metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["mean"]
                   for c in conditions]
    coarse_ci_lo = [metrics[c]["accuracy"]["mean"] - metrics[c]["accuracy"]["ci_95"][0]
                    for c in conditions]
    coarse_ci_hi = [metrics[c]["accuracy"]["ci_95"][1] - metrics[c]["accuracy"]["mean"]
                    for c in conditions]
    strict_ci_lo = [
        metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["mean"]
        - metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["ci_95"][0]
        for c in conditions
    ]
    strict_ci_hi = [
        metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["ci_95"][1]
        - metrics[c].get("strict_accuracy", metrics[c]["accuracy"])["mean"]
        for c in conditions
    ]
    x = np.arange(len(conditions))
    w = 0.35
    bars_c = ax.bar(x - w/2, coarse_accs, w,
                    color=[C_COLORS[c] for c in conditions],
                    hatch=[HATCH[c] for c in conditions],
                    edgecolor="white", linewidth=0.5,
                    yerr=[coarse_ci_lo, coarse_ci_hi],
                    error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8,
                               "ecolor": "black", "zorder": 5},
                    zorder=3, label="Category")
    bars_s = ax.bar(x + w/2, strict_accs, w,
                    color=[C_COLORS[c] for c in conditions],
                    alpha=0.4,
                    edgecolor=[C_COLORS[c] for c in conditions], linewidth=0.8,
                    yerr=[strict_ci_lo, strict_ci_hi],
                    error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8,
                               "ecolor": "black", "zorder": 5},
                    zorder=3, label="Specific")
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x); ax.set_xticklabels(["C0","C1","C2","C3"])
    ax.set_ylabel("Accuracy"); ax.set_title("(a) Decision Accuracy")
    ax.legend(fontsize=5, loc="upper left", framealpha=0.8, ncol=2)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    for bar, acc in zip(bars_c, coarse_accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.04,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=5, fontweight="bold")
    for bar, acc in zip(bars_s, strict_accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.04,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=5, color="#555555")

    # Significance brackets vs C3
    if sig and "overall_mcnemar" in sig:
        mc = sig["overall_mcnemar"]
        alpha_bf = sig.get("alpha_bonferroni", 0.0167)
        c3_x = x[3]
        for bi, cond_i in enumerate([0, 1, 2]):
            key = f"C3_vs_C{cond_i}"
            if key not in mc:
                continue
            p  = mc[key]["p_value"]
            st = _sig_stars(p, alpha_bf)
            y_br = 0.98 + bi * 0.06
            tick_h = 0.012
            ax.plot([x[cond_i], x[cond_i], c3_x, c3_x],
                    [y_br, y_br + tick_h, y_br + tick_h, y_br],
                    lw=0.7, color="black")
            ax.text((x[cond_i] + c3_x) / 2, y_br + tick_h + 0.003, st,
                    ha="center", va="bottom", fontsize=5, color="black")

    # ── Panel (b): latency stack ───────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    conds_lat = ["C1", "C2", "C3"]
    labels_b  = ["C1", "C2", "C3"]
    retrieval = [metrics[c]["latency"][c]["retrieval_ms"] for c in conds_lat]
    inference = [metrics[c]["latency"][c]["inference_ms"] for c in conds_lat]
    totals    = [metrics[c]["latency"][c]["total_ms"]     for c in conds_lat]
    overhead  = [t - r - i for t, r, i in zip(totals, retrieval, inference)]
    x = np.arange(len(conds_lat))
    ax.bar(x, retrieval, 0.5, label="Retrieval", color="#4e9dc0", hatch="////",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x, inference, 0.5, bottom=retrieval, label="Inference",
           color="#2ca02c", edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x, overhead, 0.5,
           bottom=[r+i for r, i in zip(retrieval, inference)],
           label="Overhead", color="#d6d6d6", hatch="....",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels_b)
    ax.set_ylabel("ms"); ax.set_title("(b) Latency Breakdown")
    ax.legend(fontsize=5.5, loc="upper left", framealpha=0.8)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    for xi, t in zip(x, totals):
        ax.text(xi, t + 40, f"{t:.0f}", ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    # ── Panel (c): heatmap (strict acc., coarse in parentheses) ─
    ax = fig.add_subplot(gs[0, 2])
    stypes = ["reorder", "reroute", "path", "carrier"]
    slabels = ["Reorder", "Reroute", "Path", "Carrier"]
    strict_mat = np.array([
        [metrics["by_type"][t][c].get("strict_accuracy", metrics["by_type"][t][c]["accuracy"])
         for c in conditions]
        for t in stypes
    ])
    coarse_mat = np.array([
        [metrics["by_type"][t][c]["accuracy"] for c in conditions]
        for t in stypes
    ])
    im = ax.imshow(strict_mat, aspect="auto", cmap="YlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(["C0","C1","C2","C3"], fontsize=5.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(slabels, fontsize=5.5)
    ax.set_title("(c) Specific Acc. by Type\n(cat. acc. in parens)", fontsize=6.5)
    for i in range(4):
        for j in range(4):
            sv = strict_mat[i, j]
            cv = coarse_mat[i, j]
            color = "white" if sv > 0.6 else "black"
            ax.text(j, i - 0.12, f"{sv:.2f}", ha="center", va="center", fontsize=5.5,
                    color=color, fontweight="bold")
            if abs(sv - cv) > 0.01:
                ax.text(j, i + 0.28, f"({cv:.2f})", ha="center", va="center",
                        fontsize=4.5, color=color, alpha=0.75)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # ── Panel (d): k-ablation ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax2 = ax.twinx()
    k_vals = sorted(k_data.keys())
    k_accs, k_taus, k_lats = [], [], []
    for k in k_vals:
        recs = k_data[k]
        correct = sum(1 for r in recs if _norm(r["ground_truth"]) == _norm(r["decision"]))
        k_accs.append(correct / len(recs) if recs else 0)
        k_taus.append(np.mean([r["trace_rate"] for r in recs]) if recs else 0)
        k_lats.append(np.mean([r["latency_ms"] for r in recs]) if recs else 0)
    l1, = ax.plot(k_vals, k_accs, "s-",  color="#2ca02c", label="Accuracy")
    l2, = ax.plot(k_vals, k_taus, "^--", color="#1f77b4", label="Trace τ")
    l3, = ax2.plot(k_vals, k_lats, "o:", color="#d62728", label="Latency", alpha=0.75)
    ax.axvline(8, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylim(0.7, 1.02); ax2.set_ylim(2500, 4500)
    ax.set_xticks(k_vals); ax.set_xlabel("k (retrieved records)")
    ax.set_ylabel("Acc / τ"); ax2.set_ylabel("ms", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=6)
    ax.set_title("(d) k-Ablation (Metric 8)")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend([l1, l2, l3], ["Accuracy", "Trace τ", "Latency"],
              fontsize=5.5, loc="lower right", framealpha=0.85)

    # ── Panel (e): prompt variation ────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax2 = ax.twinx()
    variants  = ["v1", "v2", "v3"]
    v_labels  = ["v1\n(Structured)", "v2\n(Step-by-step)", "v3\n(Concise)"]
    pv_accs, pv_taus, pv_lats = [], [], []
    for v in variants:
        recs = pv_data.get(v, [])
        if recs:
            correct = sum(1 for r in recs if _norm(r["ground_truth"]) == _norm(r["decision"]))
            pv_accs.append(correct / len(recs))
            pv_taus.append(np.mean([r["trace_rate"] for r in recs]))
            pv_lats.append(np.mean([r["latency_ms"] for r in recs]))
        else:
            pv_accs.append(0); pv_taus.append(0); pv_lats.append(0)
    xv = np.arange(len(variants)); w = 0.3
    b1 = ax.bar(xv - w/2, pv_accs, w, label="Accuracy",
                color="#2ca02c", edgecolor="white", linewidth=0.5, zorder=3)
    b2 = ax.bar(xv + w/2, pv_taus, w, label="Trace τ",
                color="#1f77b4", hatch="////", edgecolor="white", linewidth=0.5, zorder=3)
    l1, = ax2.plot(xv, pv_lats, "o--", color="#d62728", label="Latency", zorder=5)
    ax.set_ylim(0.5, 1.05); ax2.set_ylim(1500, 5500)
    ax.set_xticks(xv); ax.set_xticklabels(v_labels, fontsize=5.5)
    ax.set_ylabel("Acc / τ"); ax2.set_ylabel("ms", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=6)
    ax.set_title("(e) Prompt Stability (Metric 9)")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    acc_std = np.std(pv_accs)
    ax.text(0.97, 0.06, f"σ={acc_std:.4f}", transform=ax.transAxes,
            ha="right", fontsize=5.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="lightyellow", ec="grey", alpha=0.8))
    ax.legend([b1, b2, l1], ["Accuracy", "Trace τ", "Latency"],
              fontsize=5.5, loc="upper right", framealpha=0.85)

    # ── Panel (f): Cohen's κ ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    kappas = metrics["cohens_kappa"]
    pairs  = ["C0_vs_C3", "C1_vs_C3", "C2_vs_C3"]
    k_vals_kappa = [kappas[p]["kappa"] for p in pairs]
    k_colors = [C_COLORS["C0"], C_COLORS["C1"], C_COLORS["C2"]]
    k_hatches = [HATCH["C0"], HATCH["C1"], HATCH["C2"]]
    xk = np.arange(3)
    bars = ax.bar(xk, k_vals_kappa, 0.5, color=k_colors, hatch=k_hatches,
                  edgecolor="white", linewidth=0.5, zorder=3)
    for ref, lbl in [(0.4, "Moderate"), (0.6, "Substantial"), (0.8, "Near\nperfect")]:
        ax.axhline(ref, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.text(2.6, ref + 0.01, lbl, fontsize=5.5, color="grey")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(xk); ax.set_xticklabels(["C0 vs C3", "C1 vs C3", "C2 vs C3"], fontsize=5.5)
    ax.set_ylabel("Cohen's κ"); ax.set_title("(f) Inter-Condition Agreement")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    for bar, v in zip(bars, k_vals_kappa):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                f"{v:.3f}", ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    out = OUT / "fig_combined.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Main ───────────────────────────────────────────────────────

# ── Figure 7: Tamper robustness ────────────────────────────────

def load_significance() -> dict:
    path = RESULTS / "significance.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _sig_stars(p: float, alpha_bf: float) -> str:
    """Return significance annotation string from p-value and Bonferroni alpha."""
    if p < alpha_bf / 10:
        return "***"
    elif p < alpha_bf:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def load_tamper() -> dict:
    """Returns {level: {condition: [records]}} from tamper_results.json."""
    path = RESULTS / "tamper_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        all_r = json.load(f)

    baseline_path = RESULTS / "raw_results.json"
    baseline = {}
    if baseline_path.exists():
        with open(baseline_path) as f2:
            raw = json.load(f2)
        baseline = {
            "C2": [r for r in raw if r["condition"] == "C2" and r["classification"] != "error"],
            "C3": [r for r in raw if r["condition"] == "C3" and r["classification"] != "error"],
        }

    result = {"baseline": baseline}
    for level in ["targeted", "high"]:
        result[level] = {}
        for cond in ["C2", "C3"]:
            result[level][cond] = [
                r for r in all_r
                if r["tamper_level"] == level
                and r["condition"] == cond
                and r["classification"] != "error"
            ]
    return result


def fig7_tamper_robustness(tamper_data: dict) -> None:
    if not tamper_data:
        print("  fig7: no tamper_results.json — skipping")
        return

    levels_display = ["Baseline\n(0%)", "Targeted\n(~12%)", "High\n(~50%)"]
    level_keys     = ["baseline", "targeted", "high"]
    conditions     = ["C2", "C3"]
    cond_labels    = {"C2": "C2 (SQL RAG)", "C3": "C3 (PureChain)"}
    cond_colors    = {"C2": "#ff7f0e", "C3": "#2ca02c"}
    cond_hatches   = {"C2": "----",    "C3": ""}

    accs: dict = {c: [] for c in conditions}
    detect_rates: list = []

    for lk in level_keys:
        for cond in conditions:
            recs = tamper_data.get(lk, {}).get(cond, [])
            if not recs:
                accs[cond].append(None)
                continue
            correct = sum(1 for r in recs
                          if _norm(r["ground_truth"]) == _norm(r["decision"]))
            accs[cond].append(correct / len(recs))

        # C3 detection rate (fraction with tamper_alerts > 0)
        c3r = tamper_data.get(lk, {}).get("C3", [])
        if lk == "baseline" or not c3r:
            detect_rates.append(None)
        else:
            detect_rates.append(np.mean([1 if r["tamper_alerts"] > 0 else 0 for r in c3r]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2 * 0.65, 2.5))

    # ── Left panel: accuracy vs corruption level ───────────────
    x = np.arange(len(level_keys))
    w = 0.35
    for i, cond in enumerate(conditions):
        vals = [v if v is not None else 0 for v in accs[cond]]
        offset = (i - 0.5) * w
        ax1.bar(x + offset, vals, w,
                label=cond_labels[cond],
                color=cond_colors[cond],
                hatch=cond_hatches[cond],
                edgecolor="white", linewidth=0.5, zorder=3)
        for xi, v in zip(x + offset, vals):
            if v is not None:
                ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center",
                         va="bottom", fontsize=5, fontweight="bold")

    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels_display)
    ax1.set_xlabel("Corruption Level")
    ax1.set_ylabel("Decision Accuracy")
    ax1.set_title("(g) Accuracy under Tamper Attack")
    ax1.legend(fontsize=5.5, loc="lower left", framealpha=0.85)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)

    # ── Right panel: per-type accuracy gap (targeted level) ────
    stypes  = ["reorder", "reroute", "path", "carrier"]
    slabels = ["Reorder", "Reroute", "Path", "Carrier"]
    t_data  = tamper_data.get("targeted", {})
    c2_by_type, c3_by_type, gap = [], [], []

    for st in stypes:
        c2r = [r for r in t_data.get("C2", []) if r["scenario_type"] == st]
        c3r = [r for r in t_data.get("C3", []) if r["scenario_type"] == st]
        if not c2r or not c3r:
            c2_by_type.append(0); c3_by_type.append(0); gap.append(0)
            continue
        a2 = sum(1 for r in c2r if _norm(r["ground_truth"]) == _norm(r["decision"])) / len(c2r)
        a3 = sum(1 for r in c3r if _norm(r["ground_truth"]) == _norm(r["decision"])) / len(c3r)
        c2_by_type.append(a2)
        c3_by_type.append(a3)
        gap.append(a3 - a2)

    xb = np.arange(len(stypes))
    ax2.bar(xb - w/2, c2_by_type, w, label="C2 (tampered)",
            color="#ff7f0e", hatch="----", edgecolor="white", linewidth=0.5, zorder=3)
    ax2.bar(xb + w/2, c3_by_type, w, label="C3 (verified)",
            color="#2ca02c", edgecolor="white", linewidth=0.5, zorder=3)

    for idx, (xi, g) in enumerate(zip(xb, gap)):
        if abs(g) > 0.01:
            ax2.annotate(f"{g:+.2f}", xy=(xi + w/2, c3_by_type[idx] + 0.02),
                         fontsize=5, ha="center", va="bottom",
                         color="#2ca02c" if g > 0 else "#d62728", fontweight="bold")

    ax2.set_ylim(0, 1.15)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(slabels)
    ax2.set_ylabel("Decision Accuracy")
    ax2.set_title("(h) Per-type Gap (Targeted Tamper)")
    ax2.legend(fontsize=5.5, loc="lower right", framealpha=0.85)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=0.5)
    out = OUT / "fig7_tamper_robustness.pdf"
    fig.savefig(out, bbox_inches="tight")
    out_png = OUT / "fig7_tamper_robustness.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out}")


def main() -> None:
    print("\nLoading results...")
    metrics    = load_main()
    k_data     = load_ablation_k()
    pv_data    = load_ablation_prompt()
    tamper_data = load_tamper()
    sig        = load_significance()

    print(f"\nGenerating figures → {OUT}/")
    fig1_accuracy(metrics, sig=sig)
    fig2_latency(metrics)
    fig3_type_heatmap(metrics)
    fig4_k_ablation(k_data)
    fig5_prompt_stability(pv_data)
    fig6_cohens_kappa(metrics)
    fig7_tamper_robustness(tamper_data)
    fig_combined(metrics, k_data, pv_data, sig=sig)

    print("\nDone. Files:")
    for f in sorted(OUT.glob("*.pdf")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
