# Limitations (Revision R1)

Full limitation inventory for *Content-Bound Provenance for Retrieved
Evidence in Industrial LLM Safety Agents* (IEEE IoT-J, IoT-74062-2026,
R1). Section VI-G of the manuscript summarizes these. Updated 2026-10-26.

## Data and scope

1. **Synthetic single-cell benchmark.** Every decision-level result comes
   from a simulated benchmark (one scripted trigger per run, six
   observable channels, stage labels exact by construction, one
   chemistry, no pack, cooling, or BMS). None has been shown to transfer
   to field failures.
2. **Real cells validate the record mechanism only.** The 70 CALCE and
   UConn-ILCC records test detection and reconstruction of records; they
   carry no staging labels and involve no decision.
3. **Offline window selection.** Decision times are drawn from
   [60 s, T - 300 s] using each run's full duration T, which a live
   controller does not know; near-terminal decisions and runs without an
   admissible decision time (5 of 240 sampled) are not evaluated.
4. **Scenario counts.** 235 scenarios (79 actionable) plus an independent
   490-scenario replication (175 actionable); the full held-out split was
   not evaluated with the agent.

## Model and inference

5. **Local 4B backbone.** Low urgent recall even on authentic evidence;
   reads severe-label presence rather than the state-outcome association.
6. **History-dependent greedy decoding.** Temperature-0 outputs are
   reproducible for a fixed request history, not for a fixed prompt
   (14-22% of decisions change with the order of preceding requests).
   Identical prompts share one output in all contrasts; per-decision
   analyses still inherit the churn.
7. **Prompt label.** The C2/C3/C4 template presents peers as verified
   even in C2, so the three prompts differ only in record bytes. A
   control with neutral wording (one later run, both prompts interleaved)
   shows no detectable accuracy effect (actionable 0.646 vs 0.608 at
   p = 0, 0.494 vs 0.506 at p = 0.50; both CIs include zero), but the
   label changes 28 and 27 of 79 actionable decisions. That run's C2 arm
   scores 0.646 nominally against 0.544 in the main session (cross-session
   drift), so it is used only for within-run contrasts.
8. **LLM interface role is a rationale, not a result.** Citation
   faithfulness, explanation quality, and operator usefulness are not
   evaluated.

## Security and deployment

9. **Telemetry is unauthenticated.** Falsified live telemetry bounds
   every condition; device attestation is the complementary control.
10. **Not provided:** freshness, completeness, selection integrity. Genuine
    record substitution and withholding pass undetected (measured).
11. **Single public RPC endpoint.** The honest-endpoint assumption (A3)
    is stated, not established; unanimity across endpoints is implemented
    and exercised but not deployed with independent operators, and is
    not consensus validation.
12. **No formal finality.** PoA2 builds on Clique proof of authority; the
    six-block depth is an operational confirmation policy, and the
    endpoint does not expose the sealer set.
13. **Anchoring key.** No rotation or revocation. The round-1 deployment
    used a public development key and was superseded.
14. **Fail-closed is not safe.** On a verification outage the decision
    falls back to the rule engine, which still reads reported telemetry.
15. **Privacy.** Committed payloads are readable by every participant;
    encrypted payloads are recommended for proprietary data.
16. **Signed-repository comparison.** Evaluated on the deterministic rule
    only, with one simulated operator key; a replicated multi-operator
    signed store was not built.
