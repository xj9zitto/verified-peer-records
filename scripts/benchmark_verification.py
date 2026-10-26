"""
benchmark_verification.py

Measured verification-budget benchmark against the live PureChain
mainnet, backing the substrate-choice argument with results:

  (a) handle-existence check   — eth_getTransactionReceipt
  (b) content-bound check (C4) — eth_getTransaction + calldata decode
                                 + SHA-256 digest comparison

Both are timed over the same warm web3.py session on n committed
peer-anchoring transactions from data/synthetic/v14_tx_registry.json.

Output: results/verification_benchmark.json + printed report.
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.layer2.purechain_client import PureChainClient

N = 60
REGISTRY = Path("data/synthetic/v14_tx_registry.json")
OUT = Path("results/verification_benchmark.json")


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs),
            "mean_ms": round(statistics.mean(xs), 1),
            "median_ms": round(xs[len(xs) // 2], 1),
            "p95_ms": round(xs[int(0.95 * len(xs)) - 1], 1)}


def main():
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = list(reg.values())[:N]

    client = PureChainClient()
    if not client.connect():
        sys.exit("cannot connect")

    # warm the session
    client.verify_tx_hash(entries[0]["tx_hash"])
    client.verify_content(entries[0]["tx_hash"], entries[0]["digest"])

    exist_ms, content_ms = [], []
    ok_e = ok_c = 0
    for e in entries:
        t0 = time.perf_counter()
        r = client.verify_tx_hash(e["tx_hash"])
        exist_ms.append((time.perf_counter() - t0) * 1000)
        ok_e += bool(r)

        t0 = time.perf_counter()
        r = client.verify_content(e["tx_hash"], e["digest"])
        content_ms.append((time.perf_counter() - t0) * 1000)
        ok_c += bool(r)

    report = {
        "endpoint": client.rpc_url,
        "n": len(entries),
        "existence_check": {**stats(exist_ms), "passed": ok_e},
        "content_check": {**stats(content_ms), "passed": ok_c},
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    k = 5
    med_e = report["existence_check"]["median_ms"]
    med_c = report["content_check"]["median_ms"]
    print(f"\nper-decision budget (k={k}): existence {k*med_e:.0f} ms, "
          f"content-bound {k*med_c:.0f} ms")


if __name__ == "__main__":
    main()
