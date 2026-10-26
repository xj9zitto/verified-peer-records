#!/bin/bash
# r2_campaign.sh -- round-2 agent re-run, one Ollama session (IoT-74062-2026).
# Every step resumes; rerunning the script continues where it stopped.
cd /d/VibeCodingProjects/smart-factory-llm
export PYTHONUTF8=1
PY=./.venv/Scripts/python.exe
R=results/rerun_2026-10
RUN="$PY scripts/run_v14_poisoning_experiment.py --scenarios data/synthetic/v14_scenarios.json --alpha 0.5 --physics-check off --resume"
FLAG=$R/.campaign_running
mkdir -p $R
echo $$ > $FLAG
$PY scripts/keep_awake.py $FLAG &
stamp() { echo "=== $(date -Iseconds) $*"; }

stamp "wait: anchoring campaign finished"
until grep -q "^DONEREAL" logs/anchor_campaign_2026-10-16.log; do sleep 30; done
stamp "wait: all anchors final"
$PY scripts/wait_for_finality.py data/synthetic/v14_tx_registry.json || exit 1

stamp "1 nominal C0 C1 C2 @0 (C2@0 is shared by C3d, C3o, C4)"
$RUN --conditions C0 C1 C2 --rates 0 --attack-seed 42 --run-tag rep1 --out $R/nominal.json --summary-out $R/nominal_summary.json

stamp "2 main C0 C1 C2 C4 @.15 .30 .50 seed 42"
$RUN --conditions C0 C1 C2 C4 --rates 0.15 0.30 0.50 --attack-seed 42 --run-tag rep1 --out $R/main_c0124.json --summary-out $R/main_c0124_summary.json

stamp "3 C3 both policies @.15 .30 .50 seed 42"
$RUN --conditions C3 --c3-mode both --rates 0.15 0.30 0.50 --attack-seed 42 --run-tag rep1 --out $R/c3_both.json --summary-out $R/c3_both_summary.json

stamp "4 variance: repeats of the nominal arm and of C2, C4 @0.50"
for tag in rep2 rep3; do
  $RUN --conditions C2 --rates 0 --attack-seed 42 --run-tag $tag --out $R/variance_nominal.json --summary-out $R/variance_nominal_summary.json
  $RUN --conditions C2 C4 --rates 0.50 --attack-seed 42 --run-tag $tag --out $R/variance_p50.json --summary-out $R/variance_p50_summary.json
done

stamp "5 outcome ablations: state-only (C2@0 shared by C4@0; C4 @.15-.50), shuffled (C2@0)"
$RUN --conditions C2 --rates 0 --attack-seed 42 --no-outcomes --run-tag stateonly --out $R/ablation_stateonly.json --summary-out $R/ablation_stateonly_summary.json
$RUN --conditions C4 --rates 0.15 0.30 0.50 --attack-seed 42 --no-outcomes --run-tag stateonly --out $R/ablation_stateonly.json --summary-out $R/ablation_stateonly_summary.json
$RUN --conditions C2 --rates 0 --attack-seed 42 --shuffle-outcomes --run-tag shuffled --out $R/ablation_shuffled.json --summary-out $R/ablation_shuffled_summary.json

stamp "6 boundary attacks at the agent: C0 C2 C4 @0.50"
for fam in genuine_substitution withhold; do
  $RUN --conditions C0 C2 C4 --rates 0.50 --attack-seed 42 --attack-family $fam --run-tag rep1 --out $R/boundary_$fam.json --summary-out $R/boundary_${fam}_summary.json
done

stamp "7 attack draws 43, 44: C0 C2 C4 @.15 .30 .50"
for seed in 43 44; do
  $RUN --conditions C0 C2 C4 --rates 0.15 0.30 0.50 --attack-seed $seed --run-tag rep1 --out $R/seed$seed.json --summary-out $R/seed${seed}_summary.json
done

stamp "8 independent 490-scenario set: C0 C2 @0 (shared by C4@0), C0 C2 C4 @.15 .30 .50"
X="$PY scripts/run_v14_poisoning_experiment.py --scenarios data/synthetic/v14_scenarios_expanded.json --alpha 0.5 --physics-check off --resume"
$X --conditions C0 C2 --rates 0 --attack-seed 42 --run-tag rep1 --out $R/expanded.json --summary-out $R/expanded_summary.json
$X --conditions C0 C2 C4 --rates 0.15 0.30 0.50 --attack-seed 42 --run-tag rep1 --out $R/expanded.json --summary-out $R/expanded_summary.json

stamp "campaign complete"
rm -f $FLAG
