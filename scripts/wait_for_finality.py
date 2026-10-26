"""wait_for_finality.py <registry.json> [depth] -- block until every handle
in the registry is buried at least `depth` blocks (default: the client's
MIN_CONFIRMATIONS_DEFAULT).  Prints progress every minute."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()
from src.layer2.purechain_client import PureChainClient, MIN_CONFIRMATIONS_DEFAULT

reg = json.loads(Path(sys.argv[1]).read_text())
need = int(sys.argv[2]) if len(sys.argv) > 2 else MIN_CONFIRMATIONS_DEFAULT
pc = PureChainClient()
assert pc.connect()
newest = max(int(pc.w3.eth.get_transaction_receipt(v["tx_hash"])["blockNumber"])
             for v in reg.values())
while True:
    depth = pc.w3.eth.block_number - newest + 1
    print(f"newest anchor block {newest}, depth {depth} (need {need})", flush=True)
    if depth >= need:
        break
    time.sleep(60)
