"""Shared helpers for the ufc CLI."""

import json
import sys
from pathlib import Path

import joblib

# config is top-level module (package-dir = {"": "src"})
try:
    from config import (
        BASE_DIR,
        DATASET_PATH,
        FEATURE_COLS_PATH,
        FIGHTERS_CACHE_PATH,
        FIGHTS_PATH,
        MODEL_PATH,
    )
except ImportError:
    # fallback when running as `python -m src.cli` without install
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import (
        BASE_DIR,
        DATASET_PATH,
        FEATURE_COLS_PATH,
        FIGHTERS_CACHE_PATH,
        FIGHTS_PATH,
        MODEL_PATH,
    )


def resolve_paths(args):
    """Return (fights, fighters_cache, dataset, model, features) Paths respecting overrides.

    Priority:
      --data-dir  overrides data/*.json / dataset.csv base
      --model / --features override individually
    """
    data_dir = Path(args.data_dir) if getattr(args, "data_dir", None) else None

    if data_dir:
        fights = data_dir / "fights.json"
        cache = data_dir / "fighters_cache.json"
        dataset = data_dir / "dataset.csv"
        # also handle events_index.json
    else:
        fights = Path(FIGHTS_PATH)
        cache = Path(FIGHTERS_CACHE_PATH)
        dataset = Path(DATASET_PATH)

    model = Path(args.model) if getattr(args, "model", None) else Path(MODEL_PATH)
    features = Path(args.features) if getattr(args, "features", None) else Path(FEATURE_COLS_PATH)

    return fights, cache, dataset, model, features


def get_events_path(args):
    data_dir = Path(args.data_dir) if getattr(args, "data_dir", None) else None
    if data_dir:
        return data_dir / "events_index.json"
    return Path(BASE_DIR) / "data" / "events_index.json"


def load_model_and_meta(model_path, features_path):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"error: model not found at {model_path}", file=sys.stderr)
        print("  Train a model first: ufc train  (or python src/train_model.py)", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"error loading model {model_path}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        meta = joblib.load(features_path)
    except FileNotFoundError:
        print(f"error: features meta not found at {features_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"error loading features meta {features_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return model, meta


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def setup_logging(verbose=False, quiet=False):
    # placeholder — most scripts use print/tqdm directly
    pass
