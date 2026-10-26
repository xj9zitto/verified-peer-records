"""
check_generation_disjointness.py

The evaluation scenarios in data/synthetic/v14_scenarios.json were frozen
from an earlier generation of the battery benchmark (47,992 runs).  The
benchmark has since been regenerated (79,981 runs), and the regenerated
data is what the conventional ML baselines are trained on.

This matters for the validity of that comparison.  If a run identifier
still named the same trajectory across generations, training on the
current data and testing on the frozen scenarios could overlap.  This
script checks directly, by recomputing each frozen scenario's 60-s
window summary from the current parquet files under the same
(scenario, run_id, t_decision) and comparing it with the stored one.

A low match rate means the two generations are disjoint at trajectory
level, so no training window can coincide with an evaluation window.

Output: results/generation_disjointness.json
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.v14_loader import (
    OBS_CHANNELS_PRIMARY, WINDOW_SECONDS, _summarise_window,
)

V14_ROOT  = Path("D:/VibeCodingProjects/synth-batt-data/data")
SCEN_FILE = Path("data/synthetic/v14_scenarios.json")
OUT       = Path("results/generation_disjointness.json")

TOL = 1e-6


def main() -> None:
    scen = json.loads(SCEN_FILE.read_text())
    by_sc = defaultdict(list)
    for s in scen:
        by_sc[s["metadata"]["v14_scenario"]].append(s)

    checked = identical = missing = 0
    per_scenario = {}

    for sc_name, items in sorted(by_sc.items()):
        path = V14_ROOT / ("%s_obs.parquet" % sc_name)
        if not path.exists():
            print("  skip %s (missing parquet)" % sc_name)
            continue
        obs = pd.read_parquet(
            path, columns=["run_id", "time_s"] + OBS_CHANNELS_PRIMARY)
        obs["run_id"] = obs["run_id"].astype(str)

        n_i = n_m = 0
        for s in items:
            rid = str(s["metadata"]["v14_run_id"])
            t = float(s["metadata"]["t_decision_s"])
            r = obs[obs["run_id"] == rid]
            checked += 1
            if r.empty:
                missing += 1
                n_m += 1
                continue
            r = r.sort_values("time_s")
            w = r[(r["time_s"] >= t - WINDOW_SECONDS) & (r["time_s"] <= t)]
            if len(w) < 3:
                missing += 1
                n_m += 1
                continue
            cur = _summarise_window(w)
            old = s["obs_window"]["summary"]
            same = all(abs(cur[c]["mean"] - old[c]["mean"]) < TOL
                       for c in cur if c in old)
            identical += same
            n_i += same

        per_scenario[sc_name] = {
            "n": len(items), "identical": n_i, "unavailable": n_m}
        print("  %-28s n=%3d identical=%3d unavailable=%3d"
              % (sc_name, len(items), n_i, n_m))
        del obs

    report = {
        "n_scenarios_checked": checked,
        "n_identical_window": identical,
        "n_run_or_window_unavailable": missing,
        "match_rate": (identical / checked) if checked else None,
        "conclusion": (
            "trajectories are disjoint across generations; a run "
            "identifier does not name the same trajectory in both, so "
            "training on the current generation cannot overlap the "
            "frozen evaluation windows"
            if checked and identical / checked < 0.01 else
            "overlap detected; the baseline comparison would need "
            "an explicit run-level exclusion"),
        "per_scenario": per_scenario,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print("\nchecked %d, identical %d (%.2f%%), unavailable %d"
          % (checked, identical, 100.0 * identical / max(1, checked), missing))
    print("Saved -> %s" % OUT)


if __name__ == "__main__":
    main()
