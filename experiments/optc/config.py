"""OpTC experiment configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = ROOT / "checkpoints" / "OpTC"
DATA_DIR = ROOT / "data" / "OpTC"
RESULTS_DIR = ROOT / "results"

GNN_WEIGHTS = CHECKPOINT_DIR / "gnn_temp.pth"
XGBOOST_WEIGHTS = CHECKPOINT_DIR / "xgb.pkl"
WORD2VEC_WEIGHTS = CHECKPOINT_DIR / "w2v_optc.model"
RTMODEL_WEIGHTS = CHECKPOINT_DIR / "RTmodel.pth"
EMB_STORE_PATH = DATA_DIR / "emb_store.json"

BATCH_SIZES = [1000, 5000, 10000, 150000, 200000, 250000]
