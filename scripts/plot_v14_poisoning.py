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
LABEL  = {"C3": "C3d"}    # the figure shows the deployable policy only


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


def panel(ax, recs, rates, predicate, title, show_ylabel=True):
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
                lw=1.8, ms=5.5, label=LABEL.get(cond, cond))
        ax.fill_between(rates, los, his, alpha=0.15, color=COLORS[cond])
    ax.set_xlabel("Adversarial corruption rate p", fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel("Action accuracy", fontsize=10.5)
    ax.set_xlim(min(rates) - 0.02, max(rates) + 0.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(rates)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.grid(alpha=0.3)


def panel_attack_type(ax, recs, rates, gt_filter):
    """Both predicates, split by attack type.

    Handle checking and content binding coincide on fabricated
    provenance and separate on valid-provenance substitution, which is
    the distinction the panel exists to show.
    """
    series = [("C3", "fake_hash"), ("C3", "forged_hash"),
              ("C4", "fake_hash"), ("C4", "forged_hash")]
    cols  = {("C3", "fake_hash"): "#2ca02c",
             ("C3", "forged_hash"): "#d62728",
             ("C4", "fake_hash"): "#9467bd",
             ("C4", "forged_hash"): "#8c564b"}
    marks = {"fake_hash": "D", "forged_hash": "X"}
    styles = {"C3": "--", "C4": "-"}
    for cond, atype in series:
        means, los, his = [], [], []
        # rate=0 has no attacks, skip
        rates_atype = [r for r in rates if r > 0]
        for r in rates_atype:
            a, (lo, hi), n = acc(recs, lambda x, _a=atype, _r=r, _c=cond:
                                  x["rate"] == _r and x["condition"] == _c
                                  and x.get("poisoned")
                                  and x.get("attack_type") == _a
                                  and gt_filter(x))
            means.append(a  if a  is not None else np.nan)
            los.append(lo   if lo is not None else np.nan)
            his.append(hi   if hi is not None else np.nan)
        short = "Fabricated" if atype == "fake_hash" else "Valid-prov."
        ax.plot(rates_atype, means, marker=marks[atype], color=cols[(cond, atype)],
                ls=styles[cond], lw=1.8, ms=6,
                label="%s, %s" % (LABEL.get(cond, cond), short))
        ax.fill_between(rates_atype, los, his, alpha=0.12,
                        color=cols[(cond, atype)])
    # Data occupy the lower half of this panel, so the legend goes in
    # the empty upper-left corner rather than over the curves.
    ax.legend(fontsize=8.5, frameon=False, loc="upper left",
              handlelength=1.8, labelspacing=0.25)
    ax.set_xlabel("Adversarial corruption rate p", fontsize=10.5)
    ax.set_xlim(0.10, max(rates) + 0.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([r for r in rates if r > 0])
    ax.set_title("(c) By attack type", fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.grid(alpha=0.3)


def load_rerun(folder: Path):
    """Round-2 same-session records: the nominal arm (C2 at p = 0) is
    shared by C3d and C4, whose prompts are identical there; C3 is the
    deployable policy."""
    recs = []
    for name in ("nominal.json", "main_c0124.json", "c3_both.json"):
        for r in json.loads((folder / name).read_text()):
            if r.get("run_tag", "rep1") != "rep1":
                continue
            if r["condition"] == "C3" and r.get("c3_mode") != "deployable":
                continue
            recs.append(r)
    nominal_c2 = {r["scenario_id"]: r for r in recs
                  if r["condition"] == "C2" and r["rate"] == 0}
    for r in list(nominal_c2.values()):
        recs.append(dict(r, condition="C3", c3_mode="deployable",
                         shared_nominal=True))
        recs.append(dict(r, condition="C4", shared_nominal=True))
    # identical prompts share one output, exactly as in the tables
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_rerun import share_identical
    share_identical(recs, nominal_c2)
    return recs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", type=Path, default=None,
                    help="round-2 folder, e.g. results/rerun_2026-10")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()
    if args.rerun is not None:
        recs = load_rerun(args.rerun)
        _plot(recs, args.out_dir)
        return
    recs = json.loads((RESULTS_DIR / "v14_poisoning_results.json").read_text())
    # The C3 records in that file use the oracle recovery, which also
    # restores the falsified observable header and is therefore not a
    # deployable procedure.  Replace them with the deployable run; the
    # oracle is reported in the table as an upper bound instead.
    dep_path = RESULTS_DIR / "v14_poisoning_c3_deployable_results.json"
    if dep_path.exists():
        recs = [r for r in recs if r["condition"] != "C3"]
        recs += json.loads(dep_path.read_text())
        print("using deployable C3 (oracle C3 excluded from the figure)")
    c4_path = RESULTS_DIR / "v14_poisoning_c4_results.json"
    if c4_path.exists():
        recs += json.loads(c4_path.read_text())
    _plot(recs, FIG_DIR)


def _plot(recs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = sorted({r["rate"] for r in recs})
    print(f"Loaded {len(recs)} records, rates {rates}")

    # Authored near final rendered width (0.85\textwidth) so fonts
    # print at ~8 pt in the IEEE two-column layout.
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.0), sharey=True)
    panel(axes[0], recs, rates, lambda r: True,
          "(a) All scenarios")
    panel(axes[1], recs, rates,
          lambda r: r["ground_truth"] in ("replace", "urgent"),
          "(b) Actionable subset", show_ylabel=False)
    panel_attack_type(axes[2], recs, rates,
                      lambda r: r["ground_truth"] in ("replace", "urgent"))
    # one shared condition legend above the panels: cannot occlude data
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, 1.02),
               handletextpad=0.4, columnspacing=1.4)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf = out_dir / "fig9_poisoning.pdf"
    png = out_dir / "fig9_poisoning.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    print(f"  Saved -> {pdf}")
    print(f"  Saved -> {png}")


if __name__ == "__main__":
    main()
