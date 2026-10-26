"""
run_v14_poisoning_experiment.py

Adversarial observable-channel poisoning of v14 hybrid (predict + decide)
scenarios, across all four conditions (C0/C1/C2/C3).

Threat model
────────────
Attacker compromises the BMS-to-factory data interface AND the local
peer-summary database.  PureChain is immutable.

For p% of scenarios, attacker:
  1. Falsifies the 60s observable window summary (voltage/current/
     temperature/dT/dt/SOC) toward attacker-favorable 'rosy' values
     that suppress urgent / replace actions.
  2. Tampers the verified-peer-trajectory records, with either:
       'fake_hash'  : fabricated chain_tx_hash → C3 detects, recovers
       'forged_hash': real on-chain hash, content tampered → C3
                      hash-existence check passes; LLM consumes
                      tampered peer outcomes

Conditions:
  C0  rule engine on observable summary               → no defense
  C1  LLM on observable summary, no peers             → no defense
  C2  LLM + tampered peers (no hash check)            → silently deceived
  C3  LLM + on-chain-verified peers; on tamper detect →
      fall back to canonical (untampered) scenario

Hypothesis: C0/C1/C2 collapse linearly with p; C3 stays flat (against
fake_hash) and shows a shallow slope (against forged_hash).  Headline
figure: accuracy vs p, with α=0.5 mixed attack.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from tqdm import tqdm

from src.logger_setup import setup_logger
from src.layer3.llm_agent import LLMAgent
from src.layer3.llm_agent_v14 import (
    decide_v14, V14_ACTION_TOKENS, _verify_peer_records,
    build_prompt_v14,
)
from src.layer1.v14_rule_engine import c0_decide_v14
from src.dataset.v14_attack import (
    DEFAULT_RATES, FAKE_HASH, SEED,
    is_poisoned, attack_type, poison_obs_summary,
    poison_peer_outcomes, poison_scenario,
)

setup_logger("v14_poisoning")

RESULTS_DIR  = Path("results")
SCEN_FILE    = Path("data/synthetic/v14_scenarios.json")
OUT_FILE     = RESULTS_DIR / "v14_poisoning_results.json"
SUMMARY_FILE = RESULTS_DIR / "v14_poisoning_summary.json"


# ── Deterministic poison-set + attack-type selection ──────────

# The attack itself lives in src/dataset/v14_attack.py, which imports
# nothing: every analysis that needs the attack but not the agent reuses
# it from there.  These names are kept as aliases so this runner and the
# parity test in src/tests/test_v14_attack_parity.py read unchanged.
_is_poisoned        = is_poisoned
_attack_type        = attack_type
_poison_obs_summary = poison_obs_summary
_poison_peer_outcomes = poison_peer_outcomes
_poison_scenario    = poison_scenario




# ── Per-condition runner ──────────────────────────────────────

def _run_condition(
    agent,
    scenario_orig:     Dict[str, Any],
    scenario_poisoned: Dict[str, Any],
    condition:         str,
    include_outcome:   bool = True,
    c3_mode:           str = "oracle",
    physics_check:     str = "off",
) -> Dict[str, Any]:
    if condition == "C0":
        # Rule engine reads scenario_poisoned (it has no defense)
        result = dict(c0_decide_v14(scenario_poisoned))
        result["decision_source"] = "rule_engine"
        return result

    # C1/C2/C3 — go through decide_v14, which handles C3 hash-verify
    # + canonical fallback against scenario_orig
    canonical_peers = scenario_orig.get("asset_records", [])
    result = decide_v14(
        agent, scenario_poisoned, condition=condition,
        canonical_peers=canonical_peers,
        include_outcome=include_outcome,
        c3_recovery=("drop" if c3_mode in ("deployable", "both") else "canonical"),
        physics_check=physics_check,
    )
    if condition == "C3":
        result["c3_mode"] = c3_mode
    if condition == "C3" and c3_mode == "both":
        # Both recovery policies from one deployable call.  Without a
        # detection the oracle policy builds a byte-identical prompt, so
        # its record is the deployable record relabelled; with a
        # detection it re-runs on the untampered scenario exactly as
        # --c3-mode oracle does.  Returns a list of two records.
        dep = dict(result)
        dep["c3_mode"] = "deployable"
        if result.get("decision_source") != "llm" or \
                result.get("tamper_alerts", 0) == 0:
            orc = dict(dep)
            orc["c3_mode"] = "oracle"
            orc["shared_with"] = "deployable"
            return [dep, orc]
        orc = decide_v14(agent, scenario_orig, condition="C3",
                         canonical_peers=canonical_peers,
                         include_outcome=include_outcome,
                         physics_check=physics_check)
        orc["tamper_alerts"] = result["tamper_alerts"]
        orc["recovered"] = True
        orc["c3_mode"] = "oracle"
        return [dep, orc]
    # Oracle mode only: on tamper detection, re-run against the full
    # untampered scenario, which also restores the falsified observable
    # header.  Provenance verification cannot repair live telemetry, so
    # this is an upper bound, not a deployable procedure.  The deployable
    # policy withholds the rejected peers and leaves telemetry untouched.
    if (c3_mode == "oracle" and condition == "C3"
            and result.get("tamper_alerts", 0) > 0):
        try:
            result2 = decide_v14(agent, scenario_orig, condition="C3",
                                 canonical_peers=canonical_peers,
                                 include_outcome=include_outcome,
                                 physics_check=physics_check)
            result2["tamper_alerts"] = result["tamper_alerts"]
            result2["recovered"]     = True
            result2["c3_mode"]       = "oracle"
            return result2
        except Exception as exc:
            logger.error(f"C3 recovery re-run failed: {exc}")
    return result


# ── Checkpoint helpers ────────────────────────────────────────

def _load(p: Path) -> list:
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _save(p: Path, recs: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(recs, indent=2))


def _done_key(sid, cond, rate, seed, c3_mode="", tag="", family="standard"):
    return (sid, cond, rate, seed, c3_mode, tag, family)


def _done_set(recs: list) -> set:
    return {_done_key(r["scenario_id"], r["condition"], r["rate"],
                      r.get("attack_seed", SEED),
                      r.get("c3_mode", "") if r["condition"] == "C3" else "",
                      r.get("run_tag", ""), r.get("attack_family", "standard"))
            for r in recs}


# ── GT normaliser ─────────────────────────────────────────────

def _correct(rec: Dict[str, Any]) -> bool:
    return rec.get("decision") == rec.get("ground_truth")


# ── Main study ────────────────────────────────────────────────

def run_study(
    scenarios:  List[Dict[str, Any]],
    rates:      List[float],
    conditions: List[str],
    agent:      LLMAgent,
    alpha:      float = 0.5,
    out_path:   Path  = OUT_FILE,
    resume:     bool  = False,
    include_outcome: bool = True,
    c3_mode:    str   = "oracle",
    physics_check: str = "off",
    attack_seed: int  = SEED,
    run_tag:    str   = "",
    attack_family: str = "standard",
) -> List[Dict[str, Any]]:
    existing = _load(out_path) if resume else []
    done     = _done_set(existing)
    pool = None
    if attack_family == "genuine_substitution":
        from src.dataset.v14_boundary_attack import benign_peer_pool
        pool = benign_peer_pool(scenarios)

    total = len(scenarios) * len(conditions) * len(rates)
    bar = tqdm(total=total, desc="v14_poison", unit="run", dynamic_ncols=True)

    for rate in rates:
        for sc in scenarios:
            sid      = sc["scenario_id"]
            poisoned = _is_poisoned(sid, rate, seed=attack_seed)
            if attack_family == "standard":
                atype  = (_attack_type(sid, alpha, seed=attack_seed)
                          if poisoned else "none")
                s_pois = (_poison_scenario(sc, atype) if poisoned
                          else copy.deepcopy(sc))
            else:
                from src.dataset.v14_boundary_attack import (
                    apply_genuine_substitution, apply_withhold)
                atype = attack_family if poisoned else "none"
                if not poisoned:
                    s_pois = copy.deepcopy(sc)
                elif attack_family == "genuine_substitution":
                    s_pois = apply_genuine_substitution(sc, pool, attack_seed)
                else:
                    s_pois = apply_withhold(sc)

            for cond in conditions:
                bar.update(1)
                c3m = (c3_mode if cond == "C3" and c3_mode != "both"
                       else ("deployable" if cond == "C3" else ""))
                key = _done_key(sid, cond, rate, attack_seed, c3m,
                                run_tag, attack_family)
                if key in done:
                    bar.set_postfix(skip="OK", rate=rate, cond=cond)
                    continue

                try:
                    r = _run_condition(agent, sc, s_pois, cond,
                                       include_outcome=include_outcome,
                                       c3_mode=c3_mode,
                                       physics_check=physics_check)
                except Exception as exc:
                    logger.error(f"{cond}@{rate} {sid[:8]}: {exc}")
                    r = {
                        "decision": "error", "predicted_stage": None,
                        "predicted_p300": None, "classification": "error",
                        "trace_rate": 0.0, "latency_ms": 0.0,
                        "latency_retrieval_ms": 0.0, "latency_inference_ms": 0.0,
                        "model_used": "error", "tamper_alerts": 0,
                        "condition": cond, "llm_output": str(exc),
                        "violations": [], "decision_source": "error",
                    }

                for r1 in (r if isinstance(r, list) else [r]):
                    record = {
                        "attack_seed":         attack_seed,
                        "scenario_id":         sid,
                        "scenario_type":       sc.get("scenario_type", ""),
                        "ground_truth":        sc.get("ground_truth", ""),
                        "v14_scenario":        sc.get("metadata", {}).get("v14_scenario", ""),
                        "cause_family":        sc.get("metadata", {}).get("cause_family", -1),
                        "true_failure_stage":  sc.get("metadata", {}).get("failure_stage"),
                        "true_p300":           sc.get("metadata", {}).get("prevention_label_300s"),
                        "rate":                rate,
                        "poisoned":            poisoned,
                        "attack_type":         atype,
                        "attack_family":       attack_family,
                        "run_tag":             run_tag,
                        "n_withheld":          s_pois.get("_n_withheld", 0),
                        **r1,
                    }
                    if cond == "C3" and "c3_mode" not in record:
                        record["c3_mode"] = c3m
                    existing.append(record)
                _save(out_path, existing)
                done.add(key)

                ok = "OK" if _correct(record) else "X"
                bar.set_postfix(rate=rate, cond=cond,
                                pois="Y" if poisoned else "N", acc=ok)

    bar.close()
    return existing


def summarise(records: list, rates: list,
              summary_path: Path = SUMMARY_FILE) -> dict:
    conditions = sorted({(("C3d" if r.get("c3_mode") == "deployable" else "C3o")
                          if r["condition"] == "C3" and r.get("c3_mode")
                          in ("deployable", "oracle") else r["condition"])
                         for r in records})
    summary = {"by_rate": {}, "by_rate_poisoned_only": {},
               "by_attack_type": {},
               "fail_closed": sum(1 for r in records
                                  if r.get("decision_source")
                                  == "rule_engine_failclosed"),
               "errors": sum(1 for r in records
                             if r.get("decision") == "error")}
    # C3 under both policies: summarise each policy as its own condition.
    records = [dict(r, condition=("C3d" if r.get("c3_mode") == "deployable"
                                  else "C3o"))
               if r["condition"] == "C3" and r.get("c3_mode") in
               ("deployable", "oracle") else r
               for r in records]

    print("\n" + "=" * 76)
    print("  v14 input-poisoning — accuracy vs corruption rate (alpha=0.5)")
    print("=" * 76)
    print(f"  {'rate':<6} " + "  ".join(f"{c:>9}" for c in conditions))
    print("  " + "-" * 60)

    for rate in rates:
        row_acc, row_n = {}, {}
        for cond in conditions:
            rs = [r for r in records
                  if r["rate"] == rate and r["condition"] == cond
                  and r.get("classification") != "error"]
            n = len(rs)
            c = sum(1 for r in rs if _correct(r))
            row_acc[cond] = round(c/n, 4) if n else None
            row_n[cond]   = n
        print(f"  {rate:<6.2f} " + "  ".join(
            f"{row_acc[c]:>9.3f}" if row_acc[c] is not None else "       NA"
            for c in conditions
        ))
        summary["by_rate"][f"{rate:.2f}"] = {"accuracy": row_acc, "n": row_n}

    print("\n  Poisoned scenarios only (attack effectiveness):")
    print(f"  {'rate':<6} " + "  ".join(f"{c:>9}" for c in conditions))
    print("  " + "-" * 60)
    for rate in rates:
        row_acc, row_n = {}, {}
        for cond in conditions:
            rs = [r for r in records
                  if r["rate"] == rate and r["condition"] == cond
                  and r.get("poisoned")
                  and r.get("classification") != "error"]
            n = len(rs)
            c = sum(1 for r in rs if _correct(r))
            row_acc[cond] = round(c/n, 4) if n else None
            row_n[cond]   = n
        print(f"  {rate:<6.2f} " + "  ".join(
            f"{row_acc[c]:>9.3f}" if row_acc[c] is not None else "       NA"
            for c in conditions
        ))
        summary["by_rate_poisoned_only"][f"{rate:.2f}"] = {
            "accuracy": row_acc, "n": row_n}

    # Verified conditions broken down by attack type
    for vcond in [c for c in conditions if c in ("C3", "C3d", "C3o", "C4")]:
        print(f"\n  {vcond} by attack type (poisoned scenarios only):")
        print(f"  {'rate':<6}   {'fake_hash':>14}  {'forged_hash':>14}")
        print("  " + "-" * 50)
        summary["by_attack_type"][vcond] = {}
        for rate in rates:
            if rate <= 0:
                continue
            row, sub = [f"  {rate:<6.2f} "], {}
            for atype in ("fake_hash", "forged_hash"):
                rs = [r for r in records
                      if r["rate"] == rate and r["condition"] == vcond
                      and r.get("attack_type") == atype
                      and r.get("classification") != "error"]
                n = len(rs)
                c = sum(1 for r in rs if _correct(r))
                acc = round(c/n, 4) if n else None
                row.append(f"{acc:>10.3f} (n={n})" if acc is not None else f"{'NA':>14}")
                sub[atype] = {"accuracy": acc, "n": n}
            print("  ".join(row))
            summary["by_attack_type"][vcond][f"{rate:.2f}"] = sub
    print("=" * 76)

    _save(summary_path, summary)
    print(f"  Saved -> {summary_path}")
    return summary


def _write_session_meta(args, agent) -> None:
    """Record the inference session next to the results, so runs made in
    different sessions are never mistaken for repeats of one session."""
    import datetime as _dt
    import os as _os
    import urllib.request as _ur
    meta = {"started": _dt.datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv, "model": getattr(agent, "model", None)}
    base = _os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        meta["ollama_version"] = json.loads(
            _ur.urlopen(base + "/api/version", timeout=5).read())["version"]
        req = _ur.Request(base + "/api/show",
                          data=json.dumps({"model": agent.model}).encode(),
                          headers={"Content-Type": "application/json"})
        show = json.loads(_ur.urlopen(req, timeout=10).read())
        meta["model_digest"] = (show.get("details", {}) or {}).get(
            "parent_model") or show.get("modified_at")
        meta["model_details"] = show.get("details")
    except Exception as exc:
        meta["ollama_probe_error"] = str(exc)
    pc = getattr(agent, "pc_client", None)
    if pc is not None and hasattr(pc, "policy"):
        meta["contract"] = getattr(pc, "contract_address", None)
        meta["expected_writer"] = pc.policy.expected_writer
        meta["writer_source"] = getattr(pc, "writer_source", None)
        meta["min_confirmations"] = pc.policy.min_confirmations
        meta["check_function"] = pc.policy.check_function
        meta["quorum_urls"] = getattr(pc, "quorum_urls", [])
    p = Path(str(args.out) + ".session.json")
    hist = json.loads(p.read_text()) if p.exists() else []
    hist.append(meta)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hist, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates",      nargs="+", type=float, default=DEFAULT_RATES)
    ap.add_argument("--conditions", nargs="+", default=["C0", "C1", "C2", "C3"])
    ap.add_argument("--alpha",      type=float, default=0.5)
    ap.add_argument("--limit",      type=int,   default=None)
    ap.add_argument("--out",        type=Path,  default=OUT_FILE)
    ap.add_argument("--summary-out", type=Path, default=SUMMARY_FILE,
                    help="Path to write the per-rate summary JSON.")
    ap.add_argument("--scenarios",  type=Path,  default=SCEN_FILE)
    ap.add_argument("--resume",     action="store_true")
    ap.add_argument("--attack-seed", type=int, default=SEED,
                    help="Seed for the adversary's draw.  A different "
                         "seed redraws which scenarios are poisoned and "
                         "which attack each receives, giving independent "
                         "attack instances over the same scenarios.")
    ap.add_argument("--physics-check",
                    choices=["off", "clamp", "defer", "clamp_then_defer"],
                    default="off",
                    help="Physics-consistency gate policy.  The "
                         "unrepaired action is always retained in "
                         "decision_raw, so one run with the gate on "
                         "yields both arms of the ablation, paired on "
                         "identical model outputs.")
    ap.add_argument("--c3-mode", choices=["oracle", "deployable", "both"],
                    default="oracle",
                    help="C3 recovery policy on tamper detection.  "
                         "'oracle' (default, reproduces the original run) "
                         "re-runs on the full untampered scenario, which "
                         "also repairs the falsified observable header.  "
                         "'deployable' withholds the rejected peers and "
                         "leaves the corrupted telemetry unchanged.")
    ap.add_argument("--no-outcomes", action="store_true",
                    help="State-only ablation: hide peer outcomes from the LLM "
                         "(used to isolate the effect of outcome-bearing peer "
                         "evidence from verification alone).")
    ap.add_argument("--shuffle-outcomes", action="store_true",
                    help="Shuffled-outcome control: permute peer outcome "
                         "fields across the whole corpus (seeded), breaking "
                         "the state-outcome association while preserving the "
                         "outcome distribution and prompt structure.")
    ap.add_argument("--run-tag", default="",
                    help="Label stored on every record (e.g. rep1); part of "
                         "the resume key, so repeats can share a file.")
    ap.add_argument("--attack-family", default="standard",
                    choices=["standard", "genuine_substitution", "withhold"],
                    help="standard = the combined fake/forged attack; the "
                         "other two are out-of-scope boundary attacks.")
    ap.add_argument("--allow-failclosed", action="store_true",
                    help="Run C3/C4 even if verification is unavailable "
                         "(they then fail closed to the rule engine).")
    args = ap.parse_args()

    if args.out == OUT_FILE or args.summary_out == SUMMARY_FILE:
        print("Refusing to write the published round-1 result files; "
              "pass --out and --summary-out explicitly.")
        sys.exit(2)

    scenarios = json.loads(args.scenarios.read_text())
    if args.limit:
        scenarios = scenarios[: args.limit]

    if args.shuffle_outcomes:
        import random as _random
        rng = _random.Random(SEED)
        all_out = [p["outcome"] for sc in scenarios
                   for p in sc.get("asset_records", [])]
        shuffled = all_out[:]
        rng.shuffle(shuffled)
        i = 0
        for sc in scenarios:
            for p in sc.get("asset_records", []):
                p["outcome"] = shuffled[i]
                i += 1
        print(f"  SHUFFLED-OUTCOME CONTROL: {i} peer outcomes permuted "
              f"across the corpus (seed={SEED})")

    print(f"v14 Input Poisoning  —  {len(scenarios)} scenarios x "
          f"{len(args.conditions)} conditions x {len(args.rates)} rates  "
          f"= {len(scenarios)*len(args.conditions)*len(args.rates)} runs")
    print(f"  alpha = {args.alpha}  output -> {args.out}")
    if args.no_outcomes:
        print("  STATE-ONLY ABLATION: peer outcomes hidden from LLM")

    agent = LLMAgent()
    if any(c in ("C3", "C4") for c in args.conditions):
        avail = getattr(agent.pc_client, "verification_available", None)
        if (avail is None or not avail()) and not args.allow_failclosed:
            print("Verification unavailable; C3/C4 would fail closed to "
                  "the rule engine.  Aborting (use --allow-failclosed).")
            sys.exit(3)
    _write_session_meta(args, agent)
    records = run_study(scenarios, args.rates, args.conditions, agent,
                        alpha=args.alpha, out_path=args.out, resume=args.resume,
                        include_outcome=not args.no_outcomes,
                        c3_mode=args.c3_mode,
                        physics_check=args.physics_check,
                        attack_seed=args.attack_seed,
                        run_tag=args.run_tag,
                        attack_family=args.attack_family)
    summarise(records, args.rates, summary_path=args.summary_out)


if __name__ == "__main__":
    main()
