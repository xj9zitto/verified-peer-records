"""
config.py
Central configuration for the smart factory LLM framework.
All system parameters defined here — import this module
everywhere rather than hardcoding values.
"""

from pathlib import Path

# ── Project paths ──────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
SYNTHETIC_DIR   = DATA_DIR / "synthetic"
RESULTS_DIR     = ROOT_DIR / "results"
CHROMA_DIR      = DATA_DIR / "chroma_db"

# ── PI Network topology ────────────────────────────────────────
# Nodes: S1, S2 = suppliers; H1-H4 = hubs; F = factory
PI_NODES = ["S1", "S2", "H1", "H2", "H3", "H4", "F"]

PI_HOPS = [
    # (origin, dest, transit_h, amb_min, amb_max, temp_ctrl, carriers)
    ("S1", "H1", 3, 32, 40, False,  ["K1", "K2"]),
    ("S2", "H2", 2, 20, 28, False,  ["K1", "K3"]),
    ("H1", "H3", 4, 28, 36, False,  ["K2"]),
    ("H1", "H4", 5, 28, 36, True,   ["K3"]),
    ("H2", "H3", 3, 24, 32, False,  ["K2"]),
    ("H2", "H4", 4, 24, 32, True,   ["K3"]),
    ("H3", "F",  2, 26, 34, False,  ["K1", "K2"]),
    ("H4", "F",  2, 26, 34, True,   ["K3"]),
]

PI_ROUTES = [
    ("R1", ["S1", "H1", "H3", "F"]),
    ("R2", ["S1", "H1", "H4", "F"]),
    ("R3", ["S2", "H2", "H3", "F"]),
    ("R4", ["S2", "H2", "H4", "F"]),
]

# ── Carrier definitions ────────────────────────────────────────
# K1 = standard ±10°C, K2 = standard ±8°C, K3 = temp-ctrl ±2°C
CARRIERS = {
    "K1": {"type": "standard",    "variance": 10.0, "controlled": False},
    "K2": {"type": "standard",    "variance": 8.0,  "controlled": False},
    "K3": {"type": "controlled",  "variance": 2.0,  "controlled": True},
}

# ── Physics constraint thresholds ─────────────────────────────
# Sources: NASA CALCE dataset, IEC 62133-2:2017, IEEE 1188-2005
SOC_REORDER_THRESHOLD      = 20.0   # % — reorder if SOC below this
TEMP_ROUTING_BLOCK         = 45.0   # °C — block route if exceeded
DEGRADATION_REPLACE_FLAG   = 80.0   # % — flag for replacement below this

# Degradation-adaptive temperature safety (Eq. 1 in paper)
# T_safe = TEMP_ROUTING_BLOCK - max(0, DEGRADATION_REPLACE_FLAG - d) * alpha
ALPHA_DEFAULT   = 0.2   # baseline thermal sensitivity coefficient
ALPHA_RANGE     = [0.1, 0.2, 0.3]  # sensitivity analysis

# Trajectory projection thermal factor (Eq. 2 in paper)
# T_arr = T_amb_max + V_carrier + max(0, 80 - d) * beta
BETA_DEFAULT    = 0.2
BETA_RANGE      = [0.1, 0.2, 0.3]

# ── Synthetic dataset ──────────────────────────────────────────
SCENARIO_COUNTS = {
    "reorder":   30,
    "reroute":   30,
    "path":      30,
    "carrier":   30,
}
TOTAL_SCENARIOS       = 120
MULTI_CONSTRAINT_N    = 20   # subset requiring simultaneous constraints

DEGRADATION_COHORTS = {
    "healthy":  (80.0, 100.0),
    "degraded": (70.0, 80.0),
    "critical": (50.0, 70.0),
}

# ── Experimental conditions ────────────────────────────────────
CONDITIONS = {
    "C0": "Rule-engine baseline with SQL database",
    "C1": "Ungrounded LLM (parametric memory only)",
    "C2": "LLM grounded with SQL database (15% corrupted)",
    "C3": "LLM grounded with PureChain-verified records",
}
CORRUPTION_RATE = 0.15  # fraction of C2 records deliberately corrupted

# ── RAG retrieval parameters ───────────────────────────────────
K_ASSET   = 5   # top-k Type-A (asset state) records retrieved
K_ROUTING = 3   # top-k Type-B (routing condition) records retrieved
K_ABLATION = [3, 5, 8]

# ── LLM settings ──────────────────────────────────────────────
import os as _os
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()
LLM_PRIMARY     = _os.getenv("LLM_PRIMARY",   "gemma3:4b")
LLM_SECONDARY   = _os.getenv("LLM_SECONDARY", "gemma3:4b")
OLLAMA_BASE_URL = _os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
LLM_TEMPERATURE = 0.0    # deterministic outputs for evaluation

SYSTEM_PROMPT = (
    "You are a supply chain decision agent for a smart factory. "
    "Make decisions based only on the supply records provided in the "
    "context. State all numerical values explicitly before making a "
    "recommendation."
)

# ── Evaluation settings ────────────────────────────────────────
BOOTSTRAP_SAMPLES      = 1000
CONFIDENCE_LEVEL       = 0.95
HUMAN_VERIFY_SUBSET_N  = 20   # scenarios for Cohen's kappa
PROMPT_VARIATION_N     = 30   # scenarios for prompt variation test
CONFLICT_TEST_N        = 20   # scenarios for factual conflict test

# ── Embedding model ────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL = "INFO"
