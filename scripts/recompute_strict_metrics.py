"""
recompute_strict_metrics.py
Add strict_decision labels to raw_results.json and recompute metrics.json
with both coarse_accuracy (category match) and strict_accuracy (specific label match).

Strict extraction:
  - carrier  : K1/K2/K3 extracted via regex from llm_output
  - reroute  : R1/R2/R3/R4 or no_safe_route_available from llm_output
  - path     : R1/R2/R3/R4 or no_safe_route_available from llm_output
  - reorder  : reorder / reorder_degradation_flag / hold — same as coarse for now
"""

import json
import re
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS = Path("results")

# ── Ground-truth normaliser (for coarse accuracy) ──────────────
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


# ── Strict label extractor ─────────────────────────────────────

NO_ROUTE_RE = re.compile(
    # Underscore form (LLM copies from the action menu verbatim)
    r"no_safe_route_available|no_safe_route"
    # Space / natural language forms
    r"|no\s+safe\s+route|no\s+viable\s+route|no\s+route\s+available"
    r"|all\s+routes\s+unsafe|cannot\s+route|no\s+thermally\s+safe"
    r"|no\s+route\s+is\s+safe|unsafe.*route|escalate.*human",
    re.IGNORECASE,
)

ROUTE_PATTERNS = [
    re.compile(r"reroute\s+to\s+(r[1-4])\b",                 re.IGNORECASE),
    re.compile(r"use\s+route\s+(r[1-4])\b",                  re.IGNORECASE),
    re.compile(r"route\s+(r[1-4])\b",                        re.IGNORECASE),
    re.compile(r"action\s*[:\-–]\s*(r[1-4])\b",              re.IGNORECASE),
    re.compile(r"\b(r[1-4])\s*[–—\-]\s*s[12]",              re.IGNORECASE),
    re.compile(r"\brecommend\s+(r[1-4])\b",                  re.IGNORECASE),
    re.compile(r"\b(r[1-4])\b",                              re.IGNORECASE),  # fallback
]

CARRIER_PATTERNS = [
    re.compile(r"use\s+(k[123])\b",                           re.IGNORECASE),
    re.compile(r"assign\s+(k[123])\b",                        re.IGNORECASE),
    re.compile(r"select\s+(k[123])\b",                        re.IGNORECASE),
    re.compile(r"carrier\s+(k[123])\b",                       re.IGNORECASE),
    re.compile(r"action\s*[:\-–]\s*use\s+(k[123])\b",        re.IGNORECASE),
    re.compile(r"\b(k[123])\b",                               re.IGNORECASE),  # fallback
]

REORDER_DEGRADATION_RE = re.compile(
    r"reorder.*degradation|degradation.*reorder|replace.*reorder|"
    r"degradation.*flag|reorder_degradation",
    re.IGNORECASE
)


def strict_extract(text: str, scenario_type: str, ground_truth: str) -> str | None:
    """Extract specific action label from raw LLM output."""
    if not text:
        return None

    if scenario_type == "carrier":
        for pat in CARRIER_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1).upper()
        return None

    elif scenario_type in ("reroute", "path"):
        if NO_ROUTE_RE.search(text):
            return "no_safe_route_available"
        for pat in ROUTE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1).upper()
        return None

    elif scenario_type == "reorder":
        tl = text.lower()
        # Check for degradation-based reorder first
        if REORDER_DEGRADATION_RE.search(text) and ground_truth == "reorder_degradation_flag":
            return "reorder_degradation_flag"
        if any(k in tl for k in ("reorder", "replenish", "order now", "reorder immediately")):
            # Check sub-type
            if "degradation" in tl and ground_truth == "reorder_degradation_flag":
                return "reorder_degradation_flag"
            return "reorder"
        if any(k in tl for k in ("hold", "no action", "do not reorder", "maintain")):
            return "hold"
        if NO_ROUTE_RE.search(text):
            return "no_safe_route_available"
        return None

    return None


# ── Wilson score CI ────────────────────────────────────────────

def _wilson_ci(correct: int, n: int, z: float = 1.96):
    acc = correct / n if n else 0
    denom = 1 + z**2 / n
    center = (acc + z**2 / (2 * n)) / denom
    half = z * np.sqrt(acc * (1 - acc) / n + z**2 / (4 * n**2)) / denom
    return acc, round(max(0, center - half), 4), round(min(1, center + half), 4)


# ── Main ───────────────────────────────────────────────────────

def main():
    raw_path = RESULTS / "raw_results.json"
    with open(raw_path) as f:
        data = json.load(f)

    # ── Step 1: Add strict_decision to every record ────────────
    for r in data:
        r["strict_decision"] = strict_extract(
            r.get("llm_output", ""),
            r["scenario_type"],
            r["ground_truth"],
        )

    # ── Step 2: Save patched raw_results.json ─────────────────
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Patched raw_results.json with strict_decision for {len(data)} records.")

    # ── Step 3: Recompute metrics ──────────────────────────────
    with open(RESULTS / "metrics.json") as f:
        metrics = json.load(f)

    conditions = ["C0", "C1", "C2", "C3"]
    scenario_types = ["reorder", "reroute", "path", "carrier"]

    # Overall strict accuracy per condition
    print("\n=== Strict vs Coarse Accuracy ===")
    print(f"{'Type':10s} {'Cond':4s} {'Coarse':8s} {'Strict':8s}")

    for cond in conditions:
        cond_recs = [r for r in data if r["condition"] == cond and r.get("classification") != "error"]
        n = len(cond_recs)
        if n == 0:
            continue

        coarse_correct = sum(1 for r in cond_recs if _norm(r["ground_truth"]) == _norm(r["decision"]))
        coarse_acc, lo_c, hi_c = _wilson_ci(coarse_correct, n)

        if cond == "C0":
            strict_correct = coarse_correct
        else:
            strict_correct = sum(1 for r in cond_recs if r.get("strict_decision") == r["ground_truth"])
        strict_acc, lo_s, hi_s = _wilson_ci(strict_correct, n)

        # Update metrics.json
        metrics[cond]["accuracy"]["mean"]    = round(coarse_acc, 4)
        metrics[cond]["accuracy"]["ci_95"]   = [lo_c, hi_c]
        metrics[cond]["strict_accuracy"] = {
            "mean":   round(strict_acc, 4),
            "ci_95":  [lo_s, hi_s],
        }

        print(f"{'Overall':10s} {cond:4s}  {coarse_acc:.3f}    {strict_acc:.3f}")

    # Per-type breakdown
    print()
    for stype in scenario_types:
        for cond in conditions:
            recs = [r for r in data
                    if r["scenario_type"] == stype and r["condition"] == cond
                    and r.get("classification") != "error"]
            if not recs:
                continue
            n = len(recs)
            coarse_correct = sum(1 for r in recs if _norm(r["ground_truth"]) == _norm(r["decision"]))
            coarse_acc, _, _ = _wilson_ci(coarse_correct, n)

            if cond == "C0":
                strict_correct = coarse_correct
            else:
                strict_correct = sum(1 for r in recs if r.get("strict_decision") == r["ground_truth"])
            strict_acc, lo_s, hi_s = _wilson_ci(strict_correct, n)

            # Update by_type in metrics
            if stype not in metrics["by_type"]:
                metrics["by_type"][stype] = {}
            if cond not in metrics["by_type"][stype]:
                metrics["by_type"][stype][cond] = {}
            metrics["by_type"][stype][cond]["accuracy"]         = round(coarse_acc, 4)
            metrics["by_type"][stype][cond]["strict_accuracy"]  = round(strict_acc, 4)
            metrics["by_type"][stype][cond]["n"]                = n

            delta = strict_acc - coarse_acc
            flag = " !" if abs(delta) > 0.1 else ""
            print(f"  {stype:10s} {cond:4s}  coarse={coarse_acc:.3f}  strict={strict_acc:.3f}  Δ={delta:+.3f}{flag}")

    # ── Step 4: Save updated metrics.json ─────────────────────
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved updated metrics.json.")


if __name__ == "__main__":
    main()
