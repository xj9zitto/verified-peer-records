# Verified Peer Records — Code, Data, and Supplementary Artifacts

Artifact repository for the manuscript:

> **Content-Bound Verified Peer Records for Adversarially Robust
> Battery-Safety Agents in Industrial IoT**
> M. A. Dini Rage, M. S. Khaliq, M. M. H. Somrat, S. K. Ghosh,
> D.-S. Kim, J. M. Lee — submitted to the IEEE Internet of Things Journal.

It contains the complete agent source tree, experiment runners, the
deployed smart-contract source, the scenario corpora, all raw result
artifacts behind every table and figure in the paper, and the
supplementary document (platform comparison and legacy thermal-checker
definitions).

## Repository layout

```
src/                    agent, retrieval, verification, and physics layers
  layer1/               Physical-Internet event generation + v14 rule engine
  layer2/               PureChain client, canonical serialization (v14_content.py),
                        ChromaDB index, RAG retrieval
  layer3/               LLM agent, v14 hybrid pipeline (llm_agent_v14.py),
                        physics checker, two-stage retry
scripts/                experiment runners, anchoring, analysis, plotting
contracts/              SupplyEventLogger.sol (Solidity 0.8.24)
data/synthetic/         scenario corpora + on-chain transaction registries
results/                raw results (JSON) + generated figures
supplementary/          supplementary document source (LaTeX)
LIMITATIONS.md          full limitation inventory referenced by the paper
```

## Independent on-chain verification (no key required)

All 2,082 events referenced in the paper are committed on the live
PureChain mainnet and can be verified read-only by anyone:

- RPC endpoint: `https://purechainnode.com` (chain ID `900520900520`)
- Contract: `SupplyEventLogger` at
  `0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0`
- `data/synthetic/tx_registry.json` maps the 570 supply-chain events;
  `data/synthetic/v14_tx_registry.json` maps the 337 peer-anchoring
  transactions of the content-bound condition (C4), each entry holding
  the transaction hash and the SHA-256 canonical content digest.

Example (verifies one anchored peer record end to end):

```python
from src.layer2.purechain_client import PureChainClient
import json

client = PureChainClient(); client.connect()
reg = json.load(open("data/synthetic/v14_tx_registry.json"))
key, entry = next(iter(reg.items()))
payload, digest = client.get_peer_payload(entry["tx_hash"])
print(digest == entry["digest"])   # True — committed digest matches
```

## Environment

- Python 3.11, `pip install -r requirements.txt`
- [Ollama](https://ollama.com) with `gemma3:4b` pulled
  (`ollama pull gemma3:4b`); decoding is temperature 0
- Copy `.env.example` to `.env`. For read-only verification set:
  `PURECHAIN_RPC_URL=https://purechainnode.com`,
  `PURECHAIN_CHAIN_ID=900520900520`,
  `SUPPLY_EVENT_LOGGER_ADDRESS=0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0`.
  Writing new events additionally requires an authorized key
  (`PURECHAIN_PRIVATE_KEY`); none of the reproduction commands below
  need one except the optional re-anchoring step.

## Reproducing the paper's experiments

| Paper artifact | Command |
|---|---|
| v14 baseline (Table: nominal conditions) | `python scripts/run_experiment_v14.py` |
| Poisoning sweep C0–C3 (central figure) | `python scripts/run_v14_poisoning_experiment.py` |
| Content-bound condition C4 | `python scripts/run_v14_poisoning_experiment.py --conditions C4 --out results/v14_poisoning_c4_results.json --summary-out results/v14_poisoning_c4_summary.json` |
| State-only ablation | `python scripts/run_v14_poisoning_experiment.py --conditions C3 --no-outcomes --out results/v14_poisoning_ablation_results.json` |
| Shuffled-outcome control | `python scripts/run_v14_poisoning_experiment.py --conditions C3 --shuffle-outcomes --out results/v14_poisoning_shuffled_results.json` |
| Peer anchoring (needs authorized key) | `python scripts/anchor_v14_peer_digests.py --verify` |
| C4 analysis (McNemar, decomposition) | `python scripts/analyze_c4.py` |
| Balanced metrics, trivial + peer-vote baselines, Holm | `python scripts/compute_review_metrics.py` |
| Figures | `python scripts/plot_v14_poisoning.py`, `plot_v14_baseline.py`, `plot_ablation.py`, `plot_results.py`, `plot_poisoning_curve.py` |

All sampling, poisoning, and attack-type assignments are seed-controlled
(seed 42); raw results shipped in `results/` were produced by exactly
these commands.

## Dataset

The battery-safety benchmark is published separately on IEEE DataPort:

> M. A. Dini Rage, J. N. Njoku, D.-S. Kim, J. M. Lee, "BattZitto: A
> Realism-Characterized Synthetic Dataset for Li-Ion Battery Thermal
> Runaway Detection and Forecasting," IEEE Dataport, 2026.
> DOI: [10.21227/1m7z-2z53](https://dx.doi.org/10.21227/1m7z-2z53)

The 235-scenario evaluation subset derived from it is included here as
`data/synthetic/v14_scenarios.json`.

## License

MIT — see [LICENSE](LICENSE).
