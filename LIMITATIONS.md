# Limitations

Full limitation inventory for the manuscript *"Verified Peer Trajectories:
An Online Integrity Primitive for Adversarial-Robust Battery-Safety
Decisions."* The paper's **Limitations and Future Work** section summarizes
these briefly and points here for the complete list. Each item names the
future-work direction that addresses it.

1. **Simulation-based benchmark, 235-scenario subset.** The primary
   benchmark is simulation-based and uses a stratified subset of 235
   scenarios. Larger real-world datasets would be needed to confirm the
   same conclusions under genuine battery-aging and plant-operation
   conditions. *(Addressed by: broader evaluation on operational traces.)*

2. **Local 4B-parameter LLM backend.** The backend is a local
   4B-parameter model, which limits multi-step temporal reasoning over
   retrieved peer outcomes and contributes to the nominal-condition
   accuracy gap with the rule engine (C0). Larger models may narrow this
   gap but would not remove the need for explicit evidence verification.
   *(Addressed by: stronger LLM backends with multi-hop reasoning.)*

3. **Canonical recovery modeled by restoration.** The canonical-recovery
   procedure is modeled by restoring the untampered peer set for the same
   scenario, whereas a production deployment would re-fetch committed event
   payloads from the provenance layer. *(Addressed by: content-binding
   verification with re-fetch from chain-anchored payloads.)*

4. **Single attack mixture (alpha = 0.5).** The headline poisoning
   experiments use a single fabricated/valid-provenance mixture. Adaptive
   attackers may allocate the two attack types strategically across the
   corruption budget. *(Addressed by: sweeping alpha and adding adaptive
   attackers.)*

5. **Hash-existence verification only.** The deployed verification
   interface is hash-existence based, which the attack-type decomposition
   shows is sufficient against fabricated-provenance attacks but
   insufficient against valid-provenance substitution. *(Addressed by:
   content-binding verification and multi-validator co-signing.)*
