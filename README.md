# Verified Peer Records — Code, Data, and Results (Revision R1)

Artifact for the manuscript

> **Content-Bound Provenance for Retrieved Evidence in Industrial LLM
> Safety Agents**, M. A. Dini Rage, M. S. Khaliq, M. M. H. Somrat,
> S. K. Ghosh, D.-S. Kim, J. M. Lee. IEEE Internet of Things Journal,
> manuscript IoT-74062-2026, revision R1.

**Version `v2.0.0` (R1).** Tag `v1.0.0` is the artifact of the first
submission (contract `0x9fE4…a6e0`, 2,082 events, round-1 client). It is
kept for reference; every number in the R1 manuscript comes from the
files described below.

## What changed in R1

- **Contract redeployed under a new anchoring key.** The first
  deployment had been anchored under Hardhat's default development
  account, whose key is public. The same `SupplyEventLogger.sol` was
  redeployed and all 777 records re-anchored (337 + 370 peer records,
  70 real-cell records).
- **Hardened verification client** (`src/layer2/purechain_client.py`):
  chain identifier at connection; per handle receipt, success status,
  recipient contract, sender equal to a *pinned* writer, an operational
  six-block confirmation depth, and decoding of the `logAssetState`
  call with the `V14PEER` marker. Unanimity across configured endpoints.
  Transport errors are distinguished from verification failures, and
  the agent fails closed to the rule engine when verification is
  unavailable (`src/layer3/llm_agent_v14.py`).
- **Same-session agent campaign** (`scripts/r2_campaign.sh`): every
  agent condition in one inference session; conditions with
  byte-identical prompts share one output (`scripts/analyze_rerun.py`).
- **New experiments:** signed-repository recovery baseline (T5, T5x),
  attacks outside the guarantee (genuine substitution, withholding), a
  gradient-boosted classifier consuming verified evidence, and a
  verified-label control for the C2 prompt.

## Setup

- Python 3.11, `pip install -r requirements.txt`
- Copy `.env.example` to `.env`; it already holds the public values
  below. No private key is needed for anything except re-anchoring.
- For agent runs: [Ollama](https://ollama.com) with `gemma3:4b`
  (Q4_K_M; we used Ollama 0.32.9), temperature 0.

## Independent on-chain verification (no key required)

- RPC endpoint: `https://purechainnode.com`, chain ID `900520900520`
- Contract `SupplyEventLogger`: `0xB96F110c14be274B2F2e52EC658cc97A3Fd3aEA3`
- Authorised writer (contract `owner()`): `0x32a394ef2565657C27f76Ed0EdE9EEE084955476`
- `data/synthetic/v14_tx_registry.json`: 707 peer records, identity key →
  transaction hash and SHA-256 canonical-content digest.
  `data/synthetic/real_peer_tx_registry.json`: 70 real-cell records.
- `*.contract_0x9fE4.json`: the superseded round-1 anchors, kept as a
  negative control (they must be rejected: wrong recipient, wrong writer).

```python
import json
from src.layer2.purechain_client import PureChainClient, VerificationPolicy

pc = PureChainClient(
    rpc_url="https://purechainnode.com",
    contract_address="0xB96F110c14be274B2F2e52EC658cc97A3Fd3aEA3",
    policy=VerificationPolicy(
        expected_chain_id=900520900520,
        expected_writer="0x32a394ef2565657C27f76Ed0EdE9EEE084955476"))
assert pc.connect()
reg = json.load(open("data/synthetic/v14_tx_registry.json"))
ok = sum(pc.verify_content(e["tx_hash"], e["digest"]) for e in reg.values())
print(ok, "of", len(reg))      # expected: 707 of 707
```

`python scripts/validate_endpoint_checks.py` repeats the full check with
its negative controls. It reads only the public values in `.env`, but it
also writes one control transaction if `data/synthetic/endpoint_controls.json`
does not match the contract; the released file already does.

## Where each manuscript number comes from

Unpack the released results first: `python scripts/use_r1_results.py`
(copies `results/r1/` into `results/`, decompressing `.gz` files).

| Manuscript item | Source file(s) | Produced by |
|---|---|---|
| Tables IV–IX, XI, XIII, Fig. 2, Table XIV C4 row, variance paragraph | `results/rerun_2026-10/*.json` → `rerun_analysis.json` | `scripts/r2_campaign.sh`, `scripts/analyze_rerun.py`, `scripts/plot_v14_poisoning.py --rerun` |
| Table X (gate) | `results/rerun_2026-10/checker_policy_comparison.json` | `scripts/analyze_checker_policies.py` |
| Sec. V-B, verified-label control | `results/rerun_2026-10/label_control.json`, `label_control_analysis.json` | `run_v14_poisoning_experiment.py --conditions C2 C2n`, `analyze_label_control.py` |
| Order dependence of greedy decoding | `results/rerun_2026-10/inference_determinism.json` | `scripts/test_inference_determinism.py` |
| Sec. III-B, IV-D, Table III | `endpoint_check_validation.json`, `integrity_mechanism_comparison.json` | `validate_endpoint_checks.py`, `compare_integrity_mechanisms.py` |
| Table XII, 490-set rule arms | `transfer_multiseed_analysis.json`, `transfer_expanded_analysis.json` | `run_verification_transfer.py`, `analyze_transfer_multiseed.py` |
| Sec. IV-E (T5, T5x) | `v14_recovery_baseline_analysis.json` | `run_recovery_baseline.py`, `analyze_recovery_baseline.py` |
| Table XVI | `v14_boundary_attacks_analysis.json`, `rerun_2026-10/boundary_*.json` | `run_boundary_attacks.py`, `analyze_boundary_attacks.py` |
| Table XV | `v14_predictor_fusion_adaptive_aug30.json` (+ `_aug0`, `_expanded_*`, standard-attack runs) | `run_predictor_fusion.py --attack adaptive` |
| Table XVII (real cells) | `real_data_integrity.json` | `validate_real_data_integrity.py` |

`python scripts/verify_paper_numbers_r2.py` checks 73 manuscript numbers
against these files.

## Tests

```
python src/tests/test_r2_verification.py     # verifier, quorum, fail-closed (23)
python src/tests/test_v14_attack_parity.py
python src/tests/test_v14_physics_checker.py
```

## Reproducing the agent campaign

Requires the benchmark dataset (below) for the classifier experiments,
and Ollama for the agent. Greedy decoding is reproducible for a fixed
request history but not for a fixed prompt: a re-run in a different
order changes 14–22% of individual decisions while aggregate accuracy
moves by a few points, so expect agreement in the tables, not decision
for decision. The deterministic arms (rule, classifier, chain checks)
reproduce exactly.

## Dataset

> M. A. Dini Rage, J. N. Njoku, D.-S. Kim, J. M. Lee, "BattZitto: A
> Realism-Characterized Synthetic Dataset for Li-Ion Battery Thermal
> Runaway Detection and Forecasting," IEEE Dataport, 2026.
> DOI: [10.21227/1m7z-2z53](https://dx.doi.org/10.21227/1m7z-2z53)

The evaluation subsets derived from it are included as
`data/synthetic/v14_scenarios.json` (235) and
`v14_scenarios_expanded.json` (490).

## Not included

- No private key and no `.env`. The anchoring key is held by the
  authors; everything above verifies with public values only.
- Model weights (pulled through Ollama) and the benchmark dataset.

## Limitations of the artifact

The deployment exposes one public RPC endpoint, so the
honest-endpoint assumption is stated rather than established. The
six-block depth is an operational policy; Clique-style proof of
authority has no formal finality. See `LIMITATIONS.md` and Section VI-G
of the manuscript.

## License

MIT — see [LICENSE](LICENSE).
