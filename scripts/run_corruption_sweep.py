"""
run_corruption_sweep.py
Ablation: C2 accuracy vs corruption rate, compared against fixed C3 baseline.

Sweeps C2 corruption rate over {0.05, 0.10, 0.15, 0.25, 0.40} for carrier
and path scenarios (where C2-C3 gap is largest), then plots:
  - Strict accuracy vs corruption rate for C2 (degrades) and C3 (flat)
  - C3-C2 gap curve

C3 results are reused from results/raw_results.json (corruption rate doesn't
affect on-chain verification — C3 always excludes tampered records).

Usage:
    python scripts/run_corruption_sweep.py
    python scripts/run_corruption_sweep.py --types carrier path reorder
    python scripts/run_corruption_sweep.py --rates 0.05 0.10 0.15 0.25 0.40
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from src.logger_setup import setup_logger
from src.layer2.rag_retrieval import RAGRetriever
from src.layer3.llm_agent import LLMAgent
from src.layer3.physics_checker import PhysicsChecker

setup_logger("corruption_sweep")

RESULTS      = Path("results")
SWEEP_FILE   = RESULTS / "corruption_sweep.json"
SCENARIOS    = Path("data/synthetic/scenarios.json")

DEFAULT_RATES = [0.05, 0.10, 0.15, 0.25, 0.40]
DEFAULT_TYPES = ["carrier", "path"]


# ── Strict-label extractor (mirrors recompute_strict_metrics.py) ──

NO_ROUTE_RE = re.compile(
    r"no_safe_route_available|no_safe_route"
    r"|no\s+safe\s+route|no\s+viable\s+route|no\s+route\s+available"
    r"|all\s+routes\s+unsafe|cannot\s+route|no\s+thermally\s+safe"
    r"|no\s+route\s+is\s+safe|unsafe.*route|escalate.*human",
    re.IGNORECASE,
)
ROUTE_PATTERNS = [
    re.compile(r"reroute\s+to\s+(r[1-4])\b",    re.IGNORECASE),
    re.compile(r"use\s+route\s+(r[1-4])\b",      re.IGNORECASE),
    re.compile(r"route\s+(r[1-4])\b",            re.IGNORECASE),
    re.compile(r"\b(r[1-4])\b",                  re.IGNORECASE),
]
CARRIER_PATTERNS = [
    re.compile(r"use\s+(k[123])\b",              re.IGNORECASE),
    re.compile(r"assign\s+(k[123])\b",           re.IGNORECASE),
    re.compile(r"carrier\s+(k[123])\b",          re.IGNORECASE),
    re.compile(r"\b(k[123])\b",                  re.IGNORECASE),
]


def strict_extract(text: str, scenario_type: str) -> str | None:
    if not text:
        return None
    if scenario_type == "carrier":
        for pat in CARRIER_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1).upper()
        return None
    if scenario_type in ("path", "reroute"):
        if NO_ROUTE_RE.search(text):
            return "no_safe_route_available"
        for pat in ROUTE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1).upper()
        return None
    if scenario_type == "reorder":
        tl = text.lower()
        if any(k in tl for k in ("reorder", "replenish")):
            return "reorder"
        if any(k in tl for k in ("hold", "no action")):
            return "hold"
        return None
    return None


def strict_accuracy(records):
    correct = sum(
        1 for r in records
        if r.get("strict_decision") == r["ground_truth"]
        and r.get("classification") != "error"
    )
    n = sum(1 for r in records if r.get("classification") != "error")
    return correct / n if n else 0.0


# ── Main sweep ─────────────────────────────────────────────────

def run_sweep(
    rates:      list[float],
    stypes:     list[str],
    k_routing_path: int = 6,   # must match main experiment for path
    k_routing_other: int = 3,
) -> dict:
    with open(SCENARIOS) as f:
        all_scenarios = json.load(f)
    target = [s for s in all_scenarios if s["scenario_type"] in stypes]
    print(f"Scenarios: {len(target)}  types={stypes}  rates={rates}\n")

    # Load existing C3 results (reuse — corruption rate doesn't affect C3)
    with open(RESULTS / "raw_results.json") as f:
        existing = json.load(f)
    c3_by_sid = {
        r["scenario_id"]: r for r in existing
        if r["condition"] == "C3" and r["scenario_type"] in stypes
    }
    print(f"Loaded {len(c3_by_sid)} C3 baseline records from raw_results.json\n")

    sweep: dict = {"rates": rates, "types": stypes, "by_rate": {}}

    # ── C3 baseline (fixed, computed once) ────────────────────
    c3_records = []
    for sc in target:
        r = c3_by_sid.get(sc["scenario_id"])
        if r:
            r = dict(r)
            r["strict_decision"] = strict_extract(r.get("llm_output", ""), sc["scenario_type"])
            c3_records.append(r)

    c3_acc_overall = strict_accuracy(c3_records)
    c3_by_type = {}
    for stype in stypes:
        recs = [r for r in c3_records if r["scenario_type"] == stype]
        c3_by_type[stype] = strict_accuracy(recs)
    print(f"C3 baseline  overall={c3_acc_overall:.3f}  by_type={c3_by_type}\n")

    sweep["c3_baseline"] = {
        "overall": round(c3_acc_overall, 4),
        "by_type": {k: round(v, 4) for k, v in c3_by_type.items()},
    }

    # ── C2 sweep ───────────────────────────────────────────────
    checker = PhysicsChecker()

    for rate in rates:
        print(f"{'='*56}")
        print(f"Running C2 at corruption_rate={rate:.0%} ...")

        # Fresh RAGRetriever with this corruption rate
        retriever = RAGRetriever(corruption_rate=rate)
        agent = LLMAgent(retriever=retriever, checker=checker)

        records = []
        total = len(target)
        for i, sc in enumerate(target, 1):
            sid   = sc["scenario_id"]
            stype = sc["scenario_type"]
            k_r   = k_routing_path if stype == "path" else k_routing_other
            try:
                result = agent.decide(sc, condition="C2", k_routing=k_r)
                rec = {
                    "scenario_id":   sid,
                    "scenario_type": stype,
                    "ground_truth":  sc.get("ground_truth", ""),
                    "condition":     "C2",
                    "corruption_rate": rate,
                    **result,
                }
            except Exception as exc:
                print(f"  ERROR {sid[:12]}: {exc}")
                rec = {
                    "scenario_id": sid, "scenario_type": stype,
                    "ground_truth": sc.get("ground_truth",""),
                    "condition": "C2", "corruption_rate": rate,
                    "decision": "error", "classification": "error",
                    "llm_output": str(exc),
                }
            rec["strict_decision"] = strict_extract(
                rec.get("llm_output", ""), stype
            )
            records.append(rec)

            gt  = rec["ground_truth"]
            sd  = rec.get("strict_decision") or "?"
            hit = "✓" if sd == gt else "✗"
            print(f"  [{i:3d}/{total}] {sid[:14]}  [{stype:8s}]  "
                  f"gt={gt:<30} strict={sd:<10} {hit}")

        # Compute accuracy for this rate
        acc_overall = strict_accuracy(records)
        acc_by_type = {}
        for stype in stypes:
            recs = [r for r in records if r["scenario_type"] == stype]
            acc_by_type[stype] = strict_accuracy(recs)

        gap_overall = c3_acc_overall - acc_overall   # C3−C2, positive when verification helps
        print(f"\n  rate={rate:.0%}  C2={acc_overall:.3f}  C3={c3_acc_overall:.3f}  "
              f"gap={gap_overall:+.3f}  by_type={acc_by_type}\n")

        sweep["by_rate"][str(rate)] = {
            "c2_overall":  round(acc_overall, 4),
            "c3_overall":  round(c3_acc_overall, 4),
            "gap_overall": round(gap_overall, 4),
            "c2_by_type":  {k: round(v, 4) for k, v in acc_by_type.items()},
            "c3_by_type":  {k: round(v, 4) for k, v in c3_by_type.items()},
            "n_records":   len(records),
        }

        # Checkpoint after every rate
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(SWEEP_FILE, "w") as f:
            json.dump(sweep, f, indent=2)
        print(f"  Checkpoint saved → {SWEEP_FILE}")

    return sweep


# ── Plot ───────────────────────────────────────────────────────

def plot_sweep(sweep: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    # Sort rates numerically
    rates  = sorted(float(r) for r in sweep["by_rate"])
    stypes = sweep["types"]
    xpct   = [r * 100 for r in rates]

    colors  = {"C2": "#e05c5c", "C3": "#4c8cbf"}
    markers = {"C2": "o", "C3": "s"}

    # ── Figure 1: per-type accuracy curves (one panel per type) ──
    ncols  = len(stypes)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, stype in zip(axes, stypes):
        c2v = [sweep["by_rate"][str(r)]["c2_by_type"].get(stype, 0) for r in rates]
        c3v = [sweep["by_rate"][str(r)]["c3_by_type"].get(stype, 0) for r in rates]

        ax.plot(xpct, c2v, color=colors["C2"], marker=markers["C2"],
                linewidth=2, markersize=7, label="C2 (unverified)")
        ax.plot(xpct, c3v, color=colors["C3"], marker=markers["C3"],
                linewidth=2, markersize=7, linestyle="--",
                label="C3 (PureChain-verified)")
        ax.fill_between(xpct, c2v, c3v,
                        where=[c >= u for c, u in zip(c3v, c2v)],
                        alpha=0.13, color=colors["C3"],
                        label="C3 advantage")
        ax.fill_between(xpct, c2v, c3v,
                        where=[c < u for c, u in zip(c3v, c2v)],
                        alpha=0.13, color=colors["C2"],
                        label="C2 advantage")

        ax.set_title(f"{stype.capitalize()} scenarios", fontsize=12, fontweight="bold")
        ax.set_xlabel("Corruption rate (%)", fontsize=10)
        ax.set_xticks([int(r * 100) for r in rates])
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Strict accuracy", fontsize=10)
    fig.suptitle(
        "Strict Accuracy vs. Corruption Rate — C2 vs. C3 (PureChain-Verified RAG)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out1 = RESULTS / "figures" / "corruption_sweep_bytype.pdf"
    out1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out1, bbox_inches="tight", dpi=150)
    print(f"Saved → {out1}")
    plt.close(fig)

    # ── Figure 2: overall accuracy + gap bar ──────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    c2_overall = [sweep["by_rate"][str(r)]["c2_overall"] for r in rates]
    c3_overall = [sweep["by_rate"][str(r)]["c3_overall"] for r in rates]
    gaps       = [sweep["by_rate"][str(r)]["gap_overall"] for r in rates]  # C3−C2

    ax1.plot(xpct, c2_overall, color=colors["C2"], marker="o",
             linewidth=2, markersize=7, label="C2 (unverified)")
    ax1.plot(xpct, c3_overall, color=colors["C3"], marker="s",
             linewidth=2, markersize=7, linestyle="--",
             label="C3 (PureChain-verified)")
    ax1.fill_between(xpct, c2_overall, c3_overall, alpha=0.13, color=colors["C3"])
    ax1.set_xlabel("Corruption rate (%)", fontsize=10)
    ax1.set_ylabel("Strict accuracy", fontsize=10)
    ax1.set_xticks([int(r * 100) for r in rates])
    ax1.set_ylim(-0.02, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.set_title("Overall Accuracy vs. Corruption Rate", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    bar_colors = [colors["C3"] if g >= 0 else colors["C2"] for g in gaps]
    ax2.bar(xpct, [g * 100 for g in gaps], color=bar_colors,
            alpha=0.80, width=4.0, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=1.0)
    for x, g in zip(xpct, gaps):
        ax2.text(x, g * 100 + (1.5 if g >= 0 else -2.5),
                 f"{g*100:+.0f}pp", ha="center", va="bottom", fontsize=8,
                 fontweight="bold")
    ax2.set_xlabel("Corruption rate (%)", fontsize=10)
    ax2.set_ylabel("C3 − C2 accuracy gap (pp)", fontsize=10)
    ax2.set_xticks([int(r * 100) for r in rates])
    ax2.set_title("Verification Benefit (C3 − C2)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig2.suptitle(
        "PureChain Verification Benefit Scales with Corruption Severity",
        fontsize=13, fontweight="bold",
    )
    fig2.tight_layout()
    out2 = RESULTS / "figures" / "corruption_sweep_overall.pdf"
    fig2.savefig(out2, bbox_inches="tight", dpi=150)
    print(f"Saved → {out2}")
    plt.close(fig2)


# ── Entry point ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rates", nargs="+", type=float, default=DEFAULT_RATES,
        help="Corruption rates to sweep (e.g. 0.05 0.10 0.15 0.25 0.40)"
    )
    parser.add_argument(
        "--types", nargs="+", default=DEFAULT_TYPES,
        choices=["reorder", "reroute", "path", "carrier"],
        help="Scenario types to include"
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip inference — just re-plot from existing corruption_sweep.json"
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        with open(SWEEP_FILE) as f:
            sweep = json.load(f)
    else:
        sweep = run_sweep(args.rates, args.types)

    print("\n=== Corruption Sweep Summary ===")
    print(f"{'Rate':>6}  {'C2':>7}  {'C3':>7}  {'Gap':>7}")
    for rate_str, v in sweep["by_rate"].items():
        print(f"  {float(rate_str):>5.0%}  {v['c2_overall']:>7.3f}  "
              f"{v['c3_overall']:>7.3f}  {v['gap_overall']:>+7.3f}")

    plot_sweep(sweep)
    print(f"\nSweep data saved → {SWEEP_FILE}")


if __name__ == "__main__":
    main()
