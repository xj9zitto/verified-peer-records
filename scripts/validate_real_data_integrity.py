"""
validate_real_data_integrity.py

Runs the integrity layer end to end on records built from real cells,
rather than from the synthetic benchmark.

Both reviewers asked for validation against experimental or independent
data. We cannot validate battery *decisions* on these datasets: public
cycling corpora carry no per-timestep thermal-runaway staging labels, so
there is no ground truth for the stage or the action. What we can
validate, and what the paper's primary claim now rests on, is the
*evidence-integrity mechanism*, which is independent of how the records
were produced.

Records are built from CALCE CS2 (LCO 18650, with temperature) and
UConn-ILCC (NMC/graphite 18650), anchored on the live PureChain
mainnet with the same canonical serialization and content digest used
for synthetic peers, and then subjected to the same two attacks. We
report what each predicate detects and what content binding can
reconstruct.

Output: results/real_data_integrity.json
"""

from __future__ import annotations
import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.layer2.purechain_client import PureChainClient
from src.layer2.v14_content import (
    canonical_peer_payload, peer_content_digest, peer_key,
)

REAL_ROOT = Path("D:/VibeCodingProjects/synth-batt-data/real_data")
REGISTRY = Path("data/synthetic/real_peer_tx_registry.json")
RECORDS = Path("data/synthetic/real_peer_records.json")
OUT = Path("results/real_data_integrity.json")

SEED = 42
FAKE_HASH = "deadbeef" + "f" * 56
N_PER_SOURCE = 60


# ── record construction ───────────────────────────────────────

def _calce_records(limit: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Peer-style records from CALCE CS2 cycling files."""
    files = sorted((REAL_ROOT / "CS2_8" / "CS2_8").glob("*.txt"))
    out: List[Dict[str, Any]] = []
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t", engine="python")
        except Exception:
            continue
        cols = {c.strip(): c for c in df.columns}
        need = ["Time", "mV", "mA", "Temperature", "Capacity"]
        if not all(k in cols for k in need):
            continue
        df = df[[cols[k] for k in need]].dropna()
        if len(df) < 50:
            continue
        df.columns = need
        # snapshot early in the file, outcome from the remainder
        i = len(df) // 4
        row = df.iloc[i]
        tail = df.iloc[i:]
        out.append({
            "peer_run_id": f.stem,
            "scenario": "calce_cs2_8",
            "cause_family": -1,
            "snapshot_at_s": float(row["Time"]),
            "snapshot": {
                "voltage_V": round(float(row["mV"]) / 1000.0, 4),
                "current_A": round(float(row["mA"]) / 1000.0, 4),
                "temperature_C": round(float(row["Temperature"]), 3),
                "capacity_Ah": round(float(row["Capacity"]), 5),
            },
            "outcome": {
                "temperature_max_C": round(float(tail["Temperature"].max()), 3),
                "capacity_final_Ah": round(float(tail["Capacity"].iloc[-1]), 5),
                "duration_s": round(float(tail["Time"].iloc[-1]
                                          - row["Time"]), 2),
                "observed_source": "CALCE CS2_8",
            },
        })
        if len(out) >= limit:
            break
    return out


def _uconn_records(limit: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Peer-style records from UConn-ILCC cycling files."""
    out: List[Dict[str, Any]] = []
    for cell_dir in sorted((REAL_ROOT / "uconn_nmc").glob("cell_*")):
        if not cell_dir.is_dir():
            continue
        for f in sorted(cell_dir.glob("cycling_*.csv")):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            need = ["Time (s)", "Voltage (V)", "Current (A)",
                    "Capacity (Ah)", "Cycle Number"]
            if not all(c in df.columns for c in need):
                continue
            df = df[need].dropna()
            if len(df) < 50:
                continue
            for cyc, grp in list(df.groupby("Cycle Number"))[:6]:
                if len(grp) < 20:
                    continue
                i = len(grp) // 4
                row = grp.iloc[i]
                tail = grp.iloc[i:]
                out.append({
                    "peer_run_id": "%s_cyc%s" % (cell_dir.name, int(cyc)),
                    "scenario": "uconn_ilcc",
                    "cause_family": -1,
                    "snapshot_at_s": float(row["Time (s)"]),
                    "snapshot": {
                        "voltage_V": round(float(row["Voltage (V)"]), 4),
                        "current_A": round(float(row["Current (A)"]), 4),
                        "capacity_Ah": round(float(row["Capacity (Ah)"]), 5),
                    },
                    "outcome": {
                        "capacity_final_Ah": round(
                            float(tail["Capacity (Ah)"].iloc[-1]), 5),
                        "duration_s": round(float(tail["Time (s)"].iloc[-1]
                                                  - row["Time (s)"]), 2),
                        "observed_source": "UConn-ILCC",
                    },
                })
                if len(out) >= limit:
                    return out
    return out


# ── attacks ───────────────────────────────────────────────────

def tamper(rec: Dict[str, Any], attack: str) -> Dict[str, Any]:
    """Falsify the observed outcome; handle policy depends on attack."""
    r = copy.deepcopy(rec)
    o = r["outcome"]
    if "temperature_max_C" in o:
        o["temperature_max_C"] = 25.0
    if "capacity_final_Ah" in o:
        o["capacity_final_Ah"] = round(
            float(r["snapshot"].get("capacity_Ah", 1.0)) * 1.0, 5)
    o["duration_s"] = 1.0
    if attack == "fake_hash":
        r["chain_tx_hash"] = FAKE_HASH
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=N_PER_SOURCE)
    args = ap.parse_args()

    rng = random.Random(SEED)
    print("building records from real cells...")
    recs = _calce_records(args.limit, rng) + _uconn_records(args.limit, rng)
    by_src: Dict[str, int] = {}
    for r in recs:
        by_src[r["scenario"]] = by_src.get(r["scenario"], 0) + 1
    print("  %d records: %s" % (len(recs), by_src))
    if not recs:
        raise SystemExit("no real records built")

    pc = PureChainClient()
    if not pc.connect():
        raise SystemExit("cannot connect to PureChain")

    # -- anchor -----------------------------------------------
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    n_new = 0
    for r in recs:
        k = peer_key(r)
        digest = peer_content_digest(r)
        if k in registry and registry[k].get("digest") == digest:
            r["chain_tx_hash"] = registry[k]["tx_hash"]
            continue
        snap = r["snapshot"]
        tx = pc.log_v14_peer(
            canonical_peer_payload(r), digest,
            soc_bp=0,
            temp_c=int(float(snap.get("temperature_C", 0)) * 100),
            stage=0)
        registry[k] = {"tx_hash": tx.replace("0x", ""), "digest": digest}
        r["chain_tx_hash"] = registry[k]["tx_hash"]
        n_new += 1
        if n_new % 25 == 0:
            REGISTRY.write_text(json.dumps(registry, indent=1))
            print("    anchored %d" % n_new)
    REGISTRY.write_text(json.dumps(registry, indent=1))
    RECORDS.write_text(json.dumps(recs, indent=2))
    print("  anchored %d new record(s); %d total in registry"
          % (n_new, len(registry)))

    # -- attack and verify ------------------------------------
    report: Dict[str, Any] = {
        "n_records": len(recs), "by_source": by_src,
        "n_anchored_new": n_new, "attacks": {},
    }

    print("\n%-13s %8s %10s %10s %10s"
          % ("attack", "n", "handle", "content", "recovered"))
    for attack in ("none", "fake_hash", "forged_hash"):
        handle_ok = content_ok = recovered = 0
        for r in recs:
            t = r if attack == "none" else tamper(r, attack)
            h = t.get("chain_tx_hash", "")

            # handle-existence predicate
            if pc.verify_tx_hash(h):
                handle_ok += 1
            # content-bound predicate
            if pc.verify_content(h, peer_content_digest(t)):
                content_ok += 1
            elif attack != "none":
                # reconstruct from the committed payload
                got = pc.get_peer_payload(registry[peer_key(r)]["tx_hash"])
                if got is not None:
                    payload, digest = got
                    rec = json.loads(payload)
                    if peer_content_digest(rec) == digest \
                            and peer_key(rec) == peer_key(r):
                        recovered += 1

        n = len(recs)
        detected_handle = n - handle_ok if attack != "none" else 0
        detected_content = n - content_ok if attack != "none" else 0
        report["attacks"][attack] = {
            "n": n,
            "handle_accepted": handle_ok,
            "content_accepted": content_ok,
            "handle_detected": detected_handle,
            "content_detected": detected_content,
            "recovered": recovered,
        }
        print("%-13s %8d %10d %10d %10d"
              % (attack, n, detected_handle, detected_content, recovered))

    OUT.write_text(json.dumps(report, indent=2))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
