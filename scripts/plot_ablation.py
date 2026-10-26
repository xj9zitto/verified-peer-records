"""
plot_ablation.py

State-only ablation comparison.  Two-panel figure:
  (a) Actionable-subset accuracy vs corruption rate, with outcomes
      vs state-only.
  (b) Attack-type decomposition at p=0.50 on the actionable+poisoned
      subset, with outcomes vs state-only.

Output: results/figures/fig11_ablation.{pdf,png}
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with_out = json.loads((RESULTS_DIR / "v14_poisoning_results.json").read_text())
no_out   = json.loads((RESULTS_DIR / "v14_poisoning_ablation_results.json").read_text())

with_out = [r for r in with_out if r["condition"] == "C3"]
no_out   = [r for r in no_out   if r["condition"] == "C3"]

_sh_path = RESULTS_DIR / "v14_poisoning_shuffled_results.json"
shuffled = None
if _sh_path.exists():
    shuffled = [r for r in json.loads(_sh_path.read_text())
                if r["condition"] == "C3"]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0, c - h), min(1, c + h))


def correct(r):
    return r["decision"] == r["ground_truth"]


def acc(records, predicate):
    rs = [r for r in records if predicate(r)
          and r.get("classification") != "error"]
    if not rs:
        return None, (None, None), 0
    cor = sum(1 for r in rs if correct(r))
    return cor / len(rs), wilson_ci(cor, len(rs)), len(rs)


rates = [0.0, 0.15, 0.30, 0.50]

# Panels stacked vertically: full column width each when rendered
# at \columnwidth in the two-column IEEE layout.
fig, axes = plt.subplots(2, 1, figsize=(4.4, 6.6))

ax = axes[0]
_series = [("With outcomes", with_out, "#2ca02c", "D")]
if shuffled is not None:
    _series.append(("Shuffled outcomes", shuffled, "#1f77b4", "o"))
_series.append(("State-only", no_out, "#d62728", "X"))
for label, records, colour, marker in _series:
    means, los, his = [], [], []
    for r in rates:
        a, (lo, hi), n = acc(records, lambda x, _r=r:
                              x["rate"] == _r
                              and x["ground_truth"] in ("replace", "urgent"))
        means.append(a if a is not None else np.nan)
        los.append(lo if lo is not None else np.nan)
        his.append(hi if hi is not None else np.nan)
    ax.plot(rates, means, marker=marker, color=colour, lw=2, ms=8, label=label)
    ax.fill_between(rates, los, his, alpha=0.15, color=colour)
ax.set_xlabel("Adversarial corruption rate p")
ax.set_ylabel("Action accuracy (actionable subset)")
ax.set_xlim(-0.02, 0.52)
ax.set_ylim(-0.02, 1.0)
ax.set_title("(a) Actionable subset accuracy",
             fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

# Panel B — attack-type decomposition at p=0.50 on actionable+poisoned
ax = axes[1]
attack_types = ["fake_hash", "forged_hash"]
n_at = len(attack_types)
x = np.arange(n_at)
w = 0.36

means_w, los_w, his_w = [], [], []
means_s, los_s, his_s = [], [], []
ns_w, ns_s = [], []
for atype in attack_types:
    a, (lo, hi), n = acc(with_out, lambda r, _a=atype:
                          r["rate"] == 0.50
                          and r.get("poisoned")
                          and r["ground_truth"] in ("replace", "urgent")
                          and r.get("attack_type") == _a)
    means_w.append(a or 0); los_w.append(lo or 0); his_w.append(hi or 0); ns_w.append(n)
    a, (lo, hi), n = acc(no_out, lambda r, _a=atype:
                          r["rate"] == 0.50
                          and r.get("poisoned")
                          and r["ground_truth"] in ("replace", "urgent")
                          and r.get("attack_type") == _a)
    means_s.append(a or 0); los_s.append(lo or 0); his_s.append(hi or 0); ns_s.append(n)

bars_w = ax.bar(x - w/2, means_w, w, color="#2ca02c", edgecolor="black",
                lw=0.6, alpha=0.85, label="With outcomes")
bars_s = ax.bar(x + w/2, means_s, w, color="#d62728", edgecolor="black",
                lw=0.6, alpha=0.85, label="State-only", hatch="//")

# Asymmetric errorbars (clamp to non-negative to satisfy matplotlib)
yerr_w = np.maximum(0, np.stack([
    np.array(means_w) - np.array(los_w),
    np.array(his_w)   - np.array(means_w),
]))
yerr_s = np.maximum(0, np.stack([
    np.array(means_s) - np.array(los_s),
    np.array(his_s)   - np.array(means_s),
]))
ax.errorbar(x - w/2, means_w, yerr=yerr_w,
            fmt="none", ecolor="black", capsize=3, lw=0.8)
ax.errorbar(x + w/2, means_s, yerr=yerr_s,
            fmt="none", ecolor="black", capsize=3, lw=0.8)

# Value labels
for xi, y, ni in zip(x - w/2, means_w, ns_w):
    ax.text(xi, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
for xi, y, ni in zip(x + w/2, means_s, ns_s):
    ax.text(xi, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

labels = []
for atype, nw, ns in zip(attack_types, ns_w, ns_s):
    pretty = {"fake_hash": "fabricated\nprovenance",
              "forged_hash": "valid-provenance\nsubstitution"}[atype]
    labels.append(f"{pretty}\n(n={nw})")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Action accuracy (actionable + poisoned)")
ax.set_ylim(0, 1.1)
ax.set_title("(b) Attack-type decomposition at p=0.50",
             fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

plt.tight_layout()

pdf = FIG_DIR / "fig11_ablation.pdf"
png = FIG_DIR / "fig11_ablation.png"
plt.savefig(pdf, bbox_inches="tight")
plt.savefig(png, bbox_inches="tight", dpi=200)
print(f"Saved -> {pdf}")
print(f"Saved -> {png}")
