"""
verify_paper_numbers_r2.py — round-2 manuscript (paper_journal_r1/) vs results.

Each claim is (label, result file, JSON path, expected value, tolerance,
text that must appear in the manuscript sections).  A claim fails if the
result file disagrees with the expected value or if the manuscript no
longer contains the quoted text, so drift in either direction is caught.

Usage: python scripts/verify_paper_numbers_r2.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

RES = Path("results")
RR = "rerun_2026-10/rerun_analysis.json"
LC = "rerun_2026-10/label_control_analysis.json"
TEX = "".join(p.read_text(encoding="utf-8")
              for p in sorted(Path("paper_journal_r1/sections").glob("*.tex")))
# In the public artifact the manuscript sources are absent; then only the
# values are checked against the result files.
CHECK_TEXT = bool(TEX)


def get(obj: Any, path: str) -> Any:
    for part in path.split("/"):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def mcn(rate, contrast, key):
    d = json.loads((RES / RR).read_text())
    for i, t in enumerate(d["table_viii_mcnemar"]):
        if t["rate"] == rate and t["contrast"] == contrast:
            return "table_viii_mcnemar/%d/%s" % (i, key)
    raise KeyError(contrast)


CLAIMS = [
    # Table IV
    ("C0 nominal acc", RR, "table_iv_nominal/C0/action_acc", 0.685, 5e-4, "0.685"),
    ("C1 nominal acc", RR, "table_iv_nominal/C1/action_acc", 0.294, 5e-4, "0.294"),
    ("C2 nominal acc", RR, "table_iv_nominal/C2/action_acc", 0.238, 5e-4, "0.238"),
    ("C2 nominal stage", RR, "table_iv_nominal/C2/stage_acc", 0.268, 5e-4, "0.268"),
    ("C1 nominal stage", RR, "table_iv_nominal/C1/stage_acc", 0.136, 5e-4, "0.136"),
    ("C2 false-int", RR, "table_iv_nominal/C2/false_intervention", 0.917, 5e-4, "0.917"),
    # Table VII
    ("C4 act all rates", RR, "table_vii_actionable/0.50/C4/acc", 0.544, 5e-4, "0.544"),
    ("C2 act 0.50", RR, "table_vii_actionable/0.50/C2/acc", 0.481, 5e-4, "0.481"),
    ("C3d act 0.50", RR, "table_vii_actionable/0.50/C3d/acc", 0.405, 5e-4, "0.405"),
    ("C3o act 0.50", RR, "table_vii_actionable/0.50/C3o/acc", 0.532, 5e-4, "0.532"),
    ("C0 act 0.50", RR, "table_vii_actionable/0.50/C0/acc", 0.253, 5e-4, "0.253"),
    ("C1 act 0.00", RR, "table_vii_actionable/0.00/C1/acc", 0.671, 5e-4, "0.671"),
    ("C3d act 0.15", RR, "table_vii_actionable/0.15/C3d/acc", 0.443, 5e-4, "0.443"),
    # Table VI
    ("C4 urgent 0.50", RR, "table_vi_safety/C4@0.50/urgent_recall", 0.147, 5e-4, "0.147"),
    ("C4 under 0.50", RR, "table_vi_safety/C4@0.50/under_triage", 0.128, 5e-4, "0.128"),
    ("C4 cost 0.50", RR, "table_vi_safety/C4@0.50/cost_c100", 5.33, 5e-3, "5.33"),
    ("C3d cost 0.50", RR, "table_vi_safety/C3d@0.50/cost_c100", 8.46, 5e-3, "8.46"),
    # Table V
    ("pooled 0.50", RR, "table_v_seeds/0.50/pooled/delta", 0.076, 5e-4, "+0.076"),
    ("pooled 0.50 lo", RR, "table_v_seeds/0.50/pooled/ci_low", 0.025, 5e-4, "+0.025"),
    ("pooled 0.15", RR, "table_v_seeds/0.15/pooled/delta", 0.051, 5e-4, "+0.051"),
    ("pooled 0.30", RR, "table_v_seeds/0.30/pooled/delta", 0.068, 5e-4, "+0.068"),
    # Table VIII
    ("C4-C3d b", RR, None, 11, 0, "11 &  0"),
    ("C4-C2 p", RR, None, 0.0625, 1e-4, "0.0625"),
    ("C4-C3d holm", RR, None, 0.023, 5e-4, "0.023"),
    # Table IX / decomposition
    ("C4 fab 0.50", RR, "table_ix_decomp/0.50/C4%2Ffake_hash/acc", 0.65, 5e-3, "0.65"),
    ("C4 pois act 0.50", RR, "table_ix_decomp/0.50/poisoned_actionable/C4/acc", 0.538, 5e-4, "0.538"),
    ("C2 pois act 0.50", RR, "table_ix_decomp/0.50/poisoned_actionable/C2/acc", 0.410, 5e-4, "0.410"),
    # verified-label control (C2 vs C2n), Section V-B
    ("label C2 act 0", LC, "by_rate/0.00/actionable_acc/C2", 0.646, 5e-4, "\\(0.646\\)"),
    ("label C2n act 0", LC, "by_rate/0.00/actionable_acc/C2n", 0.608, 5e-4, "\\(0.608\\)"),
    ("label C2 act .5", LC, "by_rate/0.50/actionable_acc/C2", 0.494, 5e-4, "\\(0.494\\)"),
    ("label C2n act .5", LC, "by_rate/0.50/actionable_acc/C2n", 0.506, 5e-4, "\\(0.506\\)"),
    ("label d0", LC, "by_rate/0.00/delta_actionable_C2_minus_C2n/delta", 0.038, 5e-4, "\\(+0.038\\)"),
    ("label d0 lo", LC, "by_rate/0.00/delta_actionable_C2_minus_C2n/ci_low", -0.089, 5e-4, "-0.089"),
    ("label d0 hi", LC, "by_rate/0.00/delta_actionable_C2_minus_C2n/ci_high", 0.165, 5e-4, "0.165"),
    ("label d.5", LC, "by_rate/0.50/delta_actionable_C2_minus_C2n/delta", -0.013, 5e-4, "\\(-0.013\\)"),
    ("label d.5 lo", LC, "by_rate/0.50/delta_actionable_C2_minus_C2n/ci_low", -0.127, 5e-4, "-0.127"),
    ("label d.5 hi", LC, "by_rate/0.50/delta_actionable_C2_minus_C2n/ci_high", 0.101, 5e-4, "0.101"),
    ("label flips 0", LC, "by_rate/0.00/decision_flips_actionable/flipped", 28, 0, "28 and 27 of 79"),
    ("label flips .5", LC, "by_rate/0.50/decision_flips_actionable/flipped", 27, 0, "28 and 27 of 79"),
    ("label hist 0", LC, "by_rate/0.00/history_noise_C2_vs_main_campaign_C2/flips_actionable/flipped", 16, 0, "16 and 18"),
    ("label hist .5", LC, "by_rate/0.50/history_noise_C2_vs_main_campaign_C2/flips_actionable/flipped", 18, 0, "16 and 18"),
    # variance
    ("nominal flips", RR, "variance/C2@0.00/pairwise_flips/0/flipped", 34, 0, "34"),
    ("C2 p50 flips", RR, "variance/C2@0.50/pairwise_flips/0/flipped", 52, 0, "52 and 43"),
    ("cross-round flips", RR, "variance/cross_session_round1/C2_may/vs_new_C2_nominal/flipped", 60, 0, "60 of 235"),
    # Table XI
    ("state-only 0.50", RR, "table_xi_ablation/0.50/state_only_C4", 0.152, 5e-4, "0.152"),
    ("shuffled 0.00", RR, "table_xi_ablation/0.00/shuffled_C2", 0.595, 5e-4, "0.595"),
    # Table XIII (490)
    ("490 C4-C2 0.50", RR, "table_xii_expanded/0.50/delta/delta", 0.086, 5e-4, "0.086"),
    ("490 C4 0.50", RR, "table_xii_expanded/0.50/C4", 0.600, 5e-4, "0.600"),
    ("latency median s", RR, "latency_inference_ms/median", 4822, 50, "4.8"),
    # gate
    ("gate fired", "rerun_2026-10/checker_policy_comparison.json", "policies/off/gate_fired", 744, 0, "744"),
    ("clamp urgent", "rerun_2026-10/checker_policy_comparison.json", "policies/clamp/recall/urgent", 0.787, 5e-4, "0.787"),
    ("clamp under", "rerun_2026-10/checker_policy_comparison.json", "policies/clamp/under_triage_rate", 0.031, 5e-4, "0.031"),
    # chain
    ("handles", "endpoint_check_validation.json", "n_handles", 707, 0, "707"),
    ("full check ms", "endpoint_check_validation.json", "genuine/default_full/latency_ms_mean", 23.4, 0.05, "23.4"),
    ("full check p95", "endpoint_check_validation.json", "genuine/default_full/latency_ms_p95", 26.2, 0.05, "26.2"),
    ("genuine rejected", "endpoint_check_validation.json", "genuine/default_full/rejected", 0, 0, None),
    ("unreachable accepted", "endpoint_check_validation.json", "quorum_unanimous/alias_plus_unreachable/accepted", 0, 0, None),
    ("ledger verify ms", "integrity_mechanism_comparison.json", "ledger/verify_ms/mean", 24.4, 0.05, "24.4"),
    ("ecdsa ms", "integrity_mechanism_comparison.json", "signature_verify_ms/mean", 4.69, 0.005, "4.69"),
    ("signed repo ms", "integrity_mechanism_comparison.json", "signed_repo_verify_ms/mean", 5.00, 0.005, "5.00"),
    ("real-cell content", "real_data_integrity.json", "attacks/forged_hash/content_detected", 70, 0, None),
    ("real-cell clean", "real_data_integrity.json", "attacks/none/content_accepted", 70, 0, None),
    # deterministic arms
    ("T4-T2 0.50", "transfer_multiseed_analysis.json", "paired_deltas/T4-T2@0.50/delta", 0.173, 5e-4, "0.173"),
    ("490 T4-T2 0.50", "transfer_expanded_analysis.json", "paired_deltas/T4-T2@0.50/delta", 0.115, 5e-4, "0.115"),
    ("T5=T4", "v14_recovery_baseline_analysis.json", "agreement/T5-vs-T4/n_differing_decisions", 0, 0, "18{,}800"),
    ("T5x act", "v14_recovery_baseline_analysis.json", "by_rate/0.50/T5x/actionable", 0.221, 5e-4, "0.221"),
    ("substitution T4", "v14_boundary_attacks_analysis.json", "by_family/genuine_substitution/by_rate/0.50/T4/actionable", 0.221, 5e-4, None),
    ("fusion G4", "v14_predictor_fusion_adaptive_aug30.json", "analysis/by_arm/G4@0.50/actionable_accuracy", 0.872, 5e-4, "0.872"),
    ("fusion G2", "v14_predictor_fusion_adaptive_aug30.json", "analysis/by_arm/G2@0.50/actionable_accuracy", 0.479, 5e-4, "0.479"),
    ("fusion delta", "v14_predictor_fusion_adaptive_aug30.json", "analysis/pairs/G4-G2@0.50/delta", 0.392, 5e-4, "0.392"),
    ("fusion 490 delta", "v14_predictor_fusion_expanded_adaptive_aug30.json", "analysis/pairs/G4-G2@0.50/delta", 0.355, 5e-4, "0.355"),
    ("fusion std G2", "v14_predictor_fusion_aug30.json", "analysis/by_arm/G2@0.50/actionable_accuracy", 0.851, 5e-4, "0.851"),
    ("fusion aug0", "v14_predictor_fusion_adaptive_aug0.json", "analysis/by_arm/G4@0.50/actionable_accuracy", 0.490, 5e-4, "0.490"),
]


def main() -> int:
    cache, bad = {}, 0
    for label, f, path, exp, tol, text in CLAIMS:
        if path is None:   # McNemar rows located by contrast
            key = {"C4-C3d b": (0.5, "C4-C3d", "b"),
                   "C4-C2 p": (0.5, "C4-C2", "p"),
                   "C4-C3d holm": (0.5, "C4-C3d", "p_holm")}[label]
            path = mcn(*key)
        if f not in cache:
            cache[f] = json.loads((RES / f).read_text())
        try:
            # keys that contain '/' (C4/fake_hash) are written with %2F
            obj = cache[f]
            for p in [x.replace("%2F", "/") for x in path.split("/")]:
                obj = obj[int(p)] if isinstance(obj, list) else obj[p]
            val = float(obj)
        except Exception as exc:
            print("%-22s PATH ERROR %s" % (label, exc)); bad += 1; continue
        ok = abs(val - exp) <= tol + 1e-9
        in_tex = text is None or not CHECK_TEXT or text in TEX
        if not (ok and in_tex):
            bad += 1
        print("%-22s %10.4f %10.4f  %s%s" % (label, exp, val,
              "ok" if ok else "VALUE MISMATCH",
              "" if in_tex else "  TEXT MISSING (%s)" % text))
    print("\n%d claims checked, %d failed" % (len(CLAIMS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
