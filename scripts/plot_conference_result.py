"""
plot_conference_result.py

Generates the headline result figure for the BCCA conference paper:
per-action decision accuracy of C0, C2, C3 on the 4-action supply-chain
task under fabricated-hash tampering at p=0.30, with each condition's
nominal-channel accuracy overlaid as an open marker so the reader can
read off, at a glance, whether the tampered bar reaches its nominal
mark.

The figure visualises the central claim of contribution C2: that
PureChain verification preserves nominal-band accuracy on every action
type, while the rule engine (C0) and the unverified retrieval-augmented
agent (C2) degrade in class-dependent ways.

Source: results/input_poisoning_results.json (legacy 4-action task).
Output: results/figures/fig_conference_result.{pdf,png}
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = ["reorder", "reroute", "path", "carrier"]   # data keys (scenario_type)
DISPLAY = ["resupply", "reroute", "transit lane", "escort"]  # maritime labels
CONDS   = ["C0", "C2", "C3"]
COLORS  = {"C0": "#999999", "C2": "#ff7f0e", "C3": "#2ca02c"}
HATCH   = {"C0": "//",       "C2": "--",      "C3": "\\\\"}

# Same mapping used by run_input_poisoning_experiment.py so that
# action-family labels (e.g. "R3" -> "reroute") compare correctly
# against the normalised decision strings emitted by the agent.
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
    "hold":     "hold",
    "uncertain":"uncertain",
    "error":    "error",
}


def _norm(label):
    return _GT_MAP.get(str(label).lower().strip(),
                       str(label).lower().strip())


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def correct(r):
    return _norm(r["decision"]) == _norm(r["ground_truth"])


def slice_acc(records, predicate):
    rs = [r for r in records if predicate(r)
          and r.get("classification") != "error"]
    if not rs:
        return None, (None, None), 0
    cor = sum(1 for r in rs if correct(r))
    n = len(rs)
    return cor / n, wilson_ci(cor, n), n


def main():
    recs = json.loads(
        (RESULTS_DIR / "input_poisoning_results.json").read_text()
    )

    # tampered[cond][action] = (mean, lo, hi)
    tampered = {c: {} for c in CONDS}
    nominal  = {c: {} for c in CONDS}
    for cond in CONDS:
        for act in ACTIONS:
            a, ci, _ = slice_acc(
                recs,
                lambda r, _c=cond, _a=act: r["rate"] == 0.30
                                            and r["condition"] == _c
                                            and r["scenario_type"] == _a,
            )
            tampered[cond][act] = (a or 0.0, ci[0] or 0.0, ci[1] or 0.0)
            a, _, _ = slice_acc(
                recs,
                lambda r, _c=cond, _a=act: r["rate"] == 0.0
                                            and r["condition"] == _c
                                            and r["scenario_type"] == _a,
            )
            nominal[cond][act] = a or 0.0

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(ACTIONS))
    bar_w = 0.26

    for i, cond in enumerate(CONDS):
        means = np.array([tampered[cond][a][0] for a in ACTIONS])
        los   = np.array([tampered[cond][a][1] for a in ACTIONS])
        his   = np.array([tampered[cond][a][2] for a in ACTIONS])
        yerr = np.maximum(0.0, np.stack([means - los, his - means]))
        offs = (i - 1) * bar_w
        ax.bar(x + offs, means, bar_w,
               color=COLORS[cond], hatch=HATCH[cond],
               edgecolor="black", linewidth=0.6, alpha=0.85,
               label=cond)
        ax.errorbar(x + offs, means, yerr=yerr,
                    fmt="none", ecolor="black", capsize=2, linewidth=0.7)
        # Nominal markers: open diamond above the bar at the nominal value
        nom = np.array([nominal[cond][a] for a in ACTIONS])
        ax.scatter(x + offs, nom, marker="D", s=32,
                   facecolors="white", edgecolors="black", linewidths=0.9,
                   zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY)
    ax.set_ylabel("Decision accuracy under tampering ($p{=}0.30$)")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)

    # Combined legend: conditions + nominal marker
    handles, labels = ax.get_legend_handles_labels()
    nom_handle = plt.scatter([], [], marker="D", s=32,
                             facecolors="white", edgecolors="black",
                             linewidths=0.9, label="nominal acc.")
    handles.append(nom_handle)
    labels.append("nominal acc.")
    ax.legend(handles, labels, loc="lower left", fontsize=9,
              framealpha=0.9, ncol=2)

    plt.tight_layout()
    outdirs = [FIG_DIR, Path("paper_conference/figures")]
    for d in outdirs:
        d.mkdir(parents=True, exist_ok=True)
        plt.savefig(d / "fig_conference_result.pdf", bbox_inches="tight")
        plt.savefig(d / "fig_conference_result.png", bbox_inches="tight", dpi=200)
        print(f"Saved -> {d / 'fig_conference_result.pdf'}")


if __name__ == "__main__":
    main()
