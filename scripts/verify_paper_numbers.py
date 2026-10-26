"""
verify_paper_numbers.py

Checks headline numbers quoted in the manuscript against the result
files that produced them.  There is no LaTeX toolchain on this machine,
so this stands in for the usual habit of regenerating a paper's tables
from its data: each claim is stated as (source file, JSON path,
expected value) and compared with a tolerance.

Run after any re-analysis.  A failure means the manuscript and the
results have drifted apart.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

# Round-1 claims (paper_journal_resubmission/) checked against the
# result files as they stood before the round-2 redeployment.  The
# round-2 manuscript is checked by verify_paper_numbers_r2.py.
RESULTS = Path("results/archive/pre_redeploy_2026-10")


def get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("/"):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


# (label, file, json path, expected, tolerance)
CLAIMS: List[Tuple[str, str, str, float, float]] = [
    # -- deployable C3 vs oracle (Table IV) --------------------
    ("C3 deployable actionable @0.50", "safety_metrics.json",
     "by_condition_rate/C3d@0.50/actionable_accuracy", 0.392, 0.001),
    ("C3 oracle actionable @0.50", "safety_metrics.json",
     "by_condition_rate/C3@0.50/actionable_accuracy", 0.519, 0.001),
    ("C2 actionable @0.50", "safety_metrics.json",
     "by_condition_rate/C2@0.50/actionable_accuracy", 0.456, 0.001),
    ("C4 actionable @0.50", "safety_metrics.json",
     "by_condition_rate/C4@0.50/actionable_accuracy", 0.506, 0.001),

    # -- safety metrics (Table VII) ----------------------------
    ("C4 under-triage @0.50", "safety_metrics.json",
     "by_condition_rate/C4@0.50/under_triage_rate", 0.128, 0.001),
    ("C0 under-triage @0.50", "safety_metrics.json",
     "by_condition_rate/C0@0.50/under_triage_rate", 0.174, 0.001),
    ("C4 cost @0.50", "safety_metrics.json",
     "by_condition_rate/C4@0.50/expected_cost/c_miss_100", 4.63, 0.01),
    ("C0 cost @0.50", "safety_metrics.json",
     "by_condition_rate/C0@0.50/expected_cost/c_miss_100", 9.88, 0.01),

    # -- transfer, original set (Table VIII) -------------------
    ("T4-T2 @0.50 (235 set)", "transfer_multiseed_analysis.json",
     "paired_deltas/T4-T2@0.50/delta", 0.1734, 0.002),
    ("T3-T2 @0.50 (235 set)", "transfer_multiseed_analysis.json",
     "paired_deltas/T3-T2@0.50/delta", 0.0, 1e-9),
    ("T4 urgent recall @0.50", "transfer_multiseed_analysis.json",
     "by_arm_rate/T4@0.50/urgent_recall", 0.887, 0.002),
    ("T2 urgent recall @0.50", "transfer_multiseed_analysis.json",
     "by_arm_rate/T2@0.50/urgent_recall", 0.484, 0.002),

    # -- transfer, independent 490 set -------------------------
    ("T4-T2 @0.50 (490 set)", "transfer_expanded_analysis.json",
     "paired_deltas/T4-T2@0.50/delta", 0.1211, 0.002),
    ("T3-T2 @0.50 (490 set)", "transfer_expanded_analysis.json",
     "paired_deltas/T3-T2@0.50/delta", 0.0, 1e-9),

    # -- agent on the independent set (Table XI) ---------------
    ("expanded C4-C2 @0.50", "expanded_agent_analysis.json",
     "paired/0.50/delta", 0.1143, 0.002),
    ("expanded C4 actionable @0.50", "expanded_agent_analysis.json",
     "by_condition_rate/C4@0.50/actionable_accuracy", 0.611, 0.002),
    ("expanded C2 actionable @0.50", "expanded_agent_analysis.json",
     "by_condition_rate/C2@0.50/actionable_accuracy", 0.497, 0.002),

    # -- pooled seeds on the original set ----------------------
    ("pooled C4-C2 @0.50", "seed_campaign_analysis.json",
     "pooled/0.50/delta", 0.0717, 0.002),
    ("pooled C4-C2 @0.30", "seed_campaign_analysis.json",
     "pooled/0.30/delta", 0.0464, 0.002),
    ("determinism flip rate", "seed_campaign_analysis.json",
     "determinism/flip_rate", 0.1564, 0.002),

    # -- physics gate (Table VI) -------------------------------
    ("gate fire rate", "checker_ablation.json", "gate_fire_rate",
     0.765, 0.002),
    ("clamp under-triage", "checker_policy_comparison.json",
     "policies/clamp/under_triage_rate", 0.034, 0.002),
    ("ungated under-triage", "checker_policy_comparison.json",
     "policies/off/under_triage_rate", 0.114, 0.002),
    ("clamp_then_defer replace recall", "checker_policy_comparison.json",
     "policies/clamp_then_defer/recall/replace", 0.0, 1e-9),

    # -- real-cell validation (Table XII) ----------------------
    ("real forged detected by content", "real_data_integrity.json",
     "attacks/forged_hash/content_detected", 70, 0),
    ("real forged detected by handle", "real_data_integrity.json",
     "attacks/forged_hash/handle_detected", 0, 0),
    ("real forged recovered", "real_data_integrity.json",
     "attacks/forged_hash/recovered", 70, 0),
    ("real clean false positives", "real_data_integrity.json",
     "attacks/none/content_detected", 0, 0),

    # -- temporal baselines (Table X) --------------------------
    ("GBM honest accuracy", "temporal_baselines.json",
     "models/gbm/clean/accuracy", 0.969, 0.002),
    ("GRU poisoned urgent recall", "temporal_baselines.json",
     "models/gru/poisoned/urgent_recall", 0.0, 1e-9),
    ("CNN poisoned actionable", "temporal_baselines.json",
     "models/cnn/poisoned/actionable_accuracy", 0.0, 1e-9),

    # -- ML baselines on the frozen set (Table IX) -------------
    ("hgb_cost honest accuracy", "ml_baselines.json",
     "by_rate/hgb_cost@0.00/accuracy", 0.974, 0.002),
    ("hgb_cost poisoned actionable", "ml_baselines.json",
     "by_rate/hgb_cost@0.50/actionable_accuracy", 0.494, 0.002),

    # -- integrity mechanism comparison (Table V) --------------
    ("ledger verify ms", "integrity_mechanism_comparison.json",
     "ledger/verify_ms/mean", 30.7, 6.0),
    ("signature verify ms", "integrity_mechanism_comparison.json",
     "signature_verify_ms/mean", 4.96, 1.0),
    ("unique peers anchored", "integrity_mechanism_comparison.json",
     "n_unique_peers", 337, 0),

    # -- endpoint checks ---------------------------------------
    ("genuine handles rejected", "endpoint_check_validation.json",
     "genuine_rejected", 0, 0),
    ("endpoint handles checked", "endpoint_check_validation.json",
     "n_handles", 707, 0),
    # Round-trip latency over a public RPC endpoint; load-dependent, so the
    # tolerance is wide on purpose and the manuscript quotes one decimal.
    ("endpoint check latency ms", "endpoint_check_validation.json",
     "latency_ms_mean", 15.5, 6.0),
    # R2.4: the writer check must demonstrably fire, not merely pass.
    ("writer control rejections", "endpoint_check_validation.json",
     "writer_control/rejected_under_wrong_writer", 707, 0),

    # -- R2.3 authenticated-recovery baseline (Table III, VI-B) ----
    ("T5 actionable @0.50", "v14_recovery_baseline_analysis.json",
     "by_rate/0.50/T5/actionable", 0.394, 0.002),
    ("T4 actionable @0.50", "v14_recovery_baseline_analysis.json",
     "by_rate/0.50/T4/actionable", 0.394, 0.002),
    ("T5x actionable @0.50", "v14_recovery_baseline_analysis.json",
     "by_rate/0.50/T5x/actionable", 0.221, 0.002),
    # The claim is agreement decision for decision, not matching means.
    ("T5 vs T4 differing decisions", "v14_recovery_baseline_analysis.json",
     "agreement/T5-vs-T4/n_differing_decisions", 0, 0),
    ("T5x vs T2 differing decisions", "v14_recovery_baseline_analysis.json",
     "agreement/T5x-vs-T2/n_differing_decisions", 0, 0),
    ("T5x vs T3 differing decisions", "v14_recovery_baseline_analysis.json",
     "agreement/T5x-vs-T3/n_differing_decisions", 0, 0),
    ("recovery baseline paired n", "v14_recovery_baseline_analysis.json",
     "agreement/T5-vs-T4/n_paired", 18800, 0),
    ("signed repo verify ms", "integrity_mechanism_comparison.json",
     "signed_repo_verify_ms/mean", 5.39, 2.0),

    # -- R2.5 boundary attacks (Table tab:boundary, VI-B) ----------
    ("boundary in-scope T4-T2 @0.50",
     "v14_boundary_attacks_analysis.json",
     "by_family/content_tamper/deltas/T4-T2/0.50/delta", 0.173, 0.002),
    ("boundary genuine-subst T4-T2 @0.50",
     "v14_boundary_attacks_analysis.json",
     "by_family/genuine_substitution/deltas/T4-T2/0.50/delta", 0.0, 0),
    ("boundary withhold T4-T2 @0.50",
     "v14_boundary_attacks_analysis.json",
     "by_family/withhold/deltas/T4-T2/0.50/delta", 0.0, 0),
    # A zero delta is only meaningful if the predicate truly never fired.
    ("genuine-subst rejections @0.50",
     "v14_boundary_attacks_analysis.json",
     "by_family/genuine_substitution/by_rate/0.50/T4/"
     "mean_tamper_alerts_poisoned", 0.0, 0),
    ("withhold rejections @0.50", "v14_boundary_attacks_analysis.json",
     "by_family/withhold/by_rate/0.50/T4/mean_tamper_alerts_poisoned",
     0.0, 0),
    ("in-scope rejections @0.50", "v14_boundary_attacks_analysis.json",
     "by_family/content_tamper/by_rate/0.50/T4/"
     "mean_tamper_alerts_poisoned", 4.62, 0.05),
    ("withheld peers @0.50", "v14_boundary_attacks_analysis.json",
     "by_family/withhold/by_rate/0.50/mean_n_withheld", 1.85, 0.05),

    # -- missing measurements (Table XIII) ---------------------
    ("temp channel lost: urgent recall", "missing_measurements.json",
     "channel_loss/gbm/temperature_C_measured/urgent_recall", 0.0, 1e-9),
    ("temp channel lost: actionable", "missing_measurements.json",
     "channel_loss/gbm/temperature_C_measured/actionable_accuracy",
     0.548, 0.002),
    ("50% dropout: actionable", "missing_measurements.json",
     "timestep_dropout/gbm/0.50/actionable_accuracy", 0.946, 0.002),
    ("50% dropout: urgent recall", "missing_measurements.json",
     "timestep_dropout/gbm/0.50/urgent_recall", 0.977, 0.002),

    # -- corrected McNemar (Table X) ---------------------------
    ("McNemar C4-C0 @0.50 raw p", "mcnemar_corrected.json",
     "tests/18/p", 0.0042, 0.0005),
    ("McNemar C4-C3 @0.50 raw p", "mcnemar_corrected.json",
     "tests/21/p", 0.0225, 0.001),
    ("McNemar C4-C0 @0.50 delta pp", "mcnemar_corrected.json",
     "tests/18/delta_pp", 25.3, 0.15),
    # R2.2 flagged the manuscript saying "nine" discordant pairs for the
    # C4-vs-C2 contrast when the table already printed b=11, c=7.  Pin all
    # three so the prose and the table cannot drift apart again.
    ("McNemar C4-C2 @0.50 discordant", "mcnemar_corrected.json",
     "tests/20/n_discordant", 18, 0.5),
    ("McNemar C4-C2 @0.50 b", "mcnemar_corrected.json",
     "tests/20/b", 11, 0.5),
    ("McNemar C4-C2 @0.50 c", "mcnemar_corrected.json",
     "tests/20/c", 7, 0.5),

    # -- timeliness sensitivity --------------------------------
    ("sampling 30s actionable", "timeliness_sensitivity.json",
     "sampling/30s/actionable_accuracy", 0.825, 0.003),
    ("sampling 30s urgent recall", "timeliness_sensitivity.json",
     "sampling/30s/urgent_recall", 0.965, 0.003),
    ("delay 10s urgent recall", "timeliness_sensitivity.json",
     "delay/10s/urgent_recall", 0.816, 0.003),
    ("delay 30s urgent recall", "timeliness_sensitivity.json",
     "delay/30s/urgent_recall", 0.650, 0.003),
    ("delay 30s escalated frac", "timeliness_sensitivity.json",
     "delay/30s/label_escalated_frac", 0.084, 0.002),

    # -- generation disjointness -------------------------------
    ("windows reproducible from current data",
     "generation_disjointness.json", "n_identical_window", 0, 0),
    ("windows checked", "generation_disjointness.json",
     "n_scenarios_checked", 235, 0),
]


def main() -> int:
    cache = {}
    failures = 0
    print("%-42s %10s %10s  %s" % ("claim", "paper", "results", "status"))
    for label, fname, path, expected, tol in CLAIMS:
        p = RESULTS / fname
        if fname not in cache:
            if not p.exists():
                print("%-42s %10s %10s  FILE MISSING (%s)"
                      % (label, expected, "-", fname))
                failures += 1
                cache[fname] = None
                continue
            cache[fname] = json.loads(p.read_text())
        data = cache[fname]
        if data is None:
            continue
        try:
            actual = float(get(data, path))
        except Exception as exc:
            print("%-42s %10s %10s  PATH ERROR (%s)"
                  % (label, expected, "-", exc))
            failures += 1
            continue
        ok = abs(actual - expected) <= tol
        if not ok:
            failures += 1
        print("%-42s %10.4f %10.4f  %s"
              % (label, expected, actual, "ok" if ok else "MISMATCH"))

    print("\n%d claims checked, %d mismatched" % (len(CLAIMS), failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
