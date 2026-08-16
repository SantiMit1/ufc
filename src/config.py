"""Shared paths and constants for all src scripts."""
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent

FIGHTS_PATH = BASE_DIR / "data" / "fights.json"
FIGHTERS_CACHE_PATH = BASE_DIR / "data" / "fighters_cache.json"
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "ufc_stacking_ensemble.pkl"
FEATURE_COLS_PATH = BASE_DIR / "models" / "ufc_stacking_ensemble_meta.pkl"

CUTOFF_DATE = datetime(2012, 1, 1)
ELO_K = 96
ELO_INITIAL = 1500

WEIGHT_CLASSES = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight", "Catch Weight",
]
