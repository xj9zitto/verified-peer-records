"""
test_inference_determinism.py — is a temperature-0 output a function of
the prompt alone?

Builds the nominal C2 prompts for the first N scenarios and sends them
through the agent's Ollama call in three request orders: forward,
reversed, and interleaved with an unrelated prompt (the C1 prompt of the
same scenario, as in the main campaign).  Reports how many outputs
differ between orders, with the cache reset off and on
(OLLAMA_CACHE_RESET).

Output: results/rerun_2026-10/inference_determinism.json

Result (2026-10-17, 20 prompts): outputs depend on the request order
with or without the reset (9/19/19 and 9/7/10 differing raw outputs),
so the dependence is not only on the cached prefix offset.  The reset
option was removed from llm_agent.py afterwards; rerunning this script
now exercises the no-reset path twice.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

from src.layer3.llm_agent import LLMAgent
from src.layer3.llm_agent_v14 import build_prompt_v14

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = Path("results/rerun_2026-10/inference_determinism.json")


def run(agent, prompts, order, interleave=None):
    out = {}
    for i in order:
        if interleave is not None:
            agent._call_ollama(interleave[i])
        out[i] = agent._call_ollama(prompts[i])
    return out


def main() -> None:
    scen = json.loads(Path("data/synthetic/v14_scenarios.json").read_text())[:N]
    c2 = [build_prompt_v14(s, peer_records=s["asset_records"]) for s in scen]
    c1 = [build_prompt_v14(s, peer_records=None, show_peers=False) for s in scen]
    agent = LLMAgent()
    idx = list(range(N))
    report = {"n": N}
    for reset in ("0", "1"):
        os.environ["OLLAMA_CACHE_RESET"] = reset
        fwd = run(agent, c2, idx)
        rev = run(agent, c2, idx[::-1])
        itl = run(agent, c2, idx, interleave=c1)
        diff = lambda a, b: sum(a[i] != b[i] for i in idx)
        report["reset_" + reset] = {
            "forward_vs_reversed": diff(fwd, rev),
            "forward_vs_interleaved": diff(fwd, itl),
            "reversed_vs_interleaved": diff(rev, itl),
        }
        print("reset=%s" % reset, report["reset_" + reset], flush=True)
    OUT.write_text(json.dumps(report, indent=2))
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
