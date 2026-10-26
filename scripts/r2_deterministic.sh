#!/bin/bash
# r2_deterministic.sh -- round-2 non-LLM experiments on the redeployed chain.
cd /d/VibeCodingProjects/smart-factory-llm
export PYTHONUTF8=1
PY=./.venv/Scripts/python.exe
stamp() { echo "=== $(date -Iseconds) $*"; }
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61"

stamp "wait: anchoring campaign finished"
until grep -q "^DONEREAL" logs/anchor_campaign_2026-10-16.log; do sleep 30; done
$PY scripts/wait_for_finality.py data/synthetic/v14_tx_registry.json || exit 1
$PY scripts/wait_for_finality.py data/synthetic/real_peer_tx_registry.json || exit 1

stamp "endpoint checks"
$PY scripts/validate_endpoint_checks.py
stamp "real-cell record-level validation (re-evaluated after finality)"
$PY scripts/validate_real_data_integrity.py
stamp "integrity mechanism timings"
$PY scripts/compare_integrity_mechanisms.py
stamp "signed repository rebuild"
$PY scripts/build_signed_repository.py

stamp "transfer T0-T4: 235 set, seed 42 and 20 seeds; 490 set, 20 seeds"
$PY scripts/run_verification_transfer.py --seeds 42 --out results/v14_verification_transfer_results.json
$PY scripts/run_verification_transfer.py --seeds $SEEDS --out results/v14_verification_transfer_multiseed.json
$PY scripts/run_verification_transfer.py --seeds $SEEDS --scenarios data/synthetic/v14_scenarios_expanded.json --out results/v14_transfer_expanded.json
$PY scripts/analyze_transfer_multiseed.py v14_verification_transfer_multiseed.json transfer_multiseed_analysis.json
$PY scripts/analyze_transfer_multiseed.py v14_transfer_expanded.json transfer_expanded_analysis.json

stamp "recovery baseline T5 / T5x"
$PY scripts/run_recovery_baseline.py
$PY scripts/analyze_recovery_baseline.py

stamp "boundary attacks (deterministic arms)"
$PY scripts/run_boundary_attacks.py
$PY scripts/analyze_boundary_attacks.py

stamp "predictor fusion: 235 set aug 0.30 and 0; 490 set aug 0.30"
$PY scripts/run_predictor_fusion.py --aug 0.30 --tag aug30
$PY scripts/run_predictor_fusion.py --aug 0 --tag aug0
$PY scripts/run_predictor_fusion.py --aug 0.30 --tag expanded_aug30 --scenarios data/synthetic/v14_scenarios_expanded.json

stamp "deterministic campaign complete"
