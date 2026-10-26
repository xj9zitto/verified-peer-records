"""
build_artifact_release.py -- stage the R1 artifact into the public
repository clone (../verified-peer-records), without secrets.

Copies an explicit allow-list: source, tests, scripts, the contract, the
scenario corpora and public registries (handles + digests only), and the
round-2 results.  Raw result files over 5 MB are gzip-compressed.  Then
scans every staged file for the private keys held in .env and aborts if
any appears.  Nothing is committed or pushed by this script.

Usage: python scripts/build_artifact_release.py [--dest ../verified-peer-records]
"""
from __future__ import annotations
import argparse
import gzip
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIRS = ["src", "contracts"]
SCRIPTS = "scripts"
DATA = [
    "data/synthetic/v14_scenarios.json",
    "data/synthetic/v14_scenarios_expanded.json",
    "data/synthetic/v14_tx_registry.json",
    "data/synthetic/v14_tx_registry.contract_0x9fE4.json",
    "data/synthetic/real_peer_tx_registry.json",
    "data/synthetic/real_peer_tx_registry.contract_0x9fE4.json",
    "data/synthetic/real_peer_records.json",
    "data/synthetic/v14_signed_repository.json",
    "data/synthetic/endpoint_controls.json",
    "data/synthetic/scenarios.json",
    "data/synthetic/tx_registry.json",
]
RESULTS_R1 = [
    "rerun_2026-10",                         # whole folder (agent campaign)
    "endpoint_check_validation.json",
    "integrity_mechanism_comparison.json",
    "real_data_integrity.json",
    "v14_verification_transfer_results.json",
    "v14_verification_transfer_multiseed.json",
    "v14_transfer_expanded.json",
    "transfer_multiseed_analysis.json",
    "transfer_expanded_analysis.json",
    "v14_recovery_baseline_results.json",
    "v14_recovery_baseline_analysis.json",
    "v14_boundary_attacks_results.json",
    "v14_boundary_attacks_analysis.json",
    "v14_predictor_fusion_aug30.json",
    "v14_predictor_fusion_aug0.json",
    "v14_predictor_fusion_expanded_aug30.json",
    "v14_predictor_fusion_adaptive_aug30.json",
    "v14_predictor_fusion_adaptive_aug0.json",
    "v14_predictor_fusion_expanded_adaptive_aug30.json",
    "ml_baselines.json",
    "temporal_baselines.json",
    "missing_measurements.json",
    "timeliness_sensitivity.json",
    "generation_disjointness.json",
]
SKIP = re.compile(r"(__pycache__|\.pyc$|\.env|chroma_db|node_modules)")
GZIP_ABOVE = 5 * 1024 * 1024


def copy_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        if p.is_dir() or SKIP.search(str(p)):
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size > GZIP_ABOVE:
        with open(src, "rb") as fi, gzip.open(str(dst) + ".gz", "wb", 9) as fo:
            shutil.copyfileobj(fi, fo)
    else:
        shutil.copy2(src, dst)


def secrets() -> list[str]:
    keys = []
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "PRIVATE_KEY" in line and "=" in line and not line.lstrip().startswith("#"):
            v = line.split("=", 1)[1].strip().replace("0x", "")
            if len(v) == 64:
                keys.append(v.lower())
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=ROOT.parent / "verified-peer-records")
    args = ap.parse_args()
    dst = args.dest
    assert (dst / ".git").exists(), "destination must be the artifact clone"

    for d in DIRS:
        copy_tree(ROOT / d, dst / d)
    copy_tree(ROOT / SCRIPTS, dst / SCRIPTS)
    for f in DATA:
        copy_file(ROOT / f, dst / f)
    for f in RESULTS_R1:
        src = ROOT / "results" / f
        if src.is_dir():
            copy_tree(src, dst / "results" / "r1" / f)
        else:
            copy_file(src, dst / "results" / "r1" / f)
    shutil.copy2(ROOT / "docs" / "LIMITATIONS.md", dst / "LIMITATIONS.md")
    for f in ["hardhat.config.js", "package.json", "requirements.txt"]:
        shutil.copy2(ROOT / f, dst / f)

    keys = secrets()
    leaks = []
    for p in dst.rglob("*"):
        if p.is_dir() or ".git" in p.parts:
            continue
        try:
            data = p.read_bytes()
            if p.suffix == ".gz":
                data = gzip.decompress(data)
            low = data.lower()
        except Exception:
            continue
        for k in keys:
            if k.encode() in low:
                leaks.append(str(p))
    if leaks:
        print("SECRET FOUND in:", leaks)
        return 1
    print("staged into %s; %d keys checked, no secrets found" % (dst, len(keys)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
