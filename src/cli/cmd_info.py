"""Info command — show data/model status."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from cli._common import get_events_path, resolve_paths
from config import BASE_DIR, CUTOFF_DATE


def handle_info(args):

    fights_path, cache_path, dataset_path, model_path, features_path = resolve_paths(args)
    events_path = get_events_path(args)
    as_json = args.json

    info = {
        "cutoff_date": CUTOFF_DATE.strftime("%Y-%m-%d"),
        "paths": {
            "events_index": str(events_path),
            "fights": str(fights_path),
            "fighters_cache": str(cache_path),
            "dataset": str(dataset_path),
            "model": str(model_path),
            "features": str(features_path),
        },
        "data": {},
        "model": {},
    }

    # Events index
    if events_path.exists():
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                events = json.load(f)
            info["data"]["events_index"] = {"exists": True, "count": len(events)}
            if events:
                dates = [e.get("date") for e in events if e.get("date")]
                if dates:
                    info["data"]["events_index"]["first_date"] = min(dates)
                    info["data"]["events_index"]["last_date"] = max(dates)
                scrapped = sum(1 for e in events if e.get("scrapped"))
                info["data"]["events_index"]["scrapped"] = scrapped
                info["data"]["events_index"]["pending"] = len(events) - scrapped
        except Exception as e:
            info["data"]["events_index"] = {"exists": True, "error": str(e)}
    else:
        info["data"]["events_index"] = {"exists": False}

    # Fights
    if fights_path.exists():
        try:
            with open(fights_path, "r", encoding="utf-8") as f:
                fights = json.load(f)
            info["data"]["fights"] = {"exists": True, "count": len(fights)}
            if fights:
                dates = [f.get("event_date") for f in fights if f.get("event_date")]
                if dates:
                    info["data"]["fights"]["first_date"] = min(dates)
                    info["data"]["fights"]["last_date"] = max(dates)
                # count wins etc.
                info["data"]["fights"]["sample_event"] = fights[0].get("event_name") if fights else None
        except Exception as e:
            info["data"]["fights"] = {"exists": True, "error": str(e)}
    else:
        info["data"]["fights"] = {"exists": False}

    # Fighters cache
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            # cache can be dict or list
            if isinstance(cache, dict):
                count = len(cache)
            elif isinstance(cache, list):
                count = len(cache)
            else:
                count = 0
            info["data"]["fighters_cache"] = {"exists": True, "count": count}
        except Exception as e:
            info["data"]["fighters_cache"] = {"exists": True, "error": str(e)}
    else:
        info["data"]["fighters_cache"] = {"exists": False}

    # Dataset
    if dataset_path.exists():
        try:
            df = pd.read_csv(dataset_path, nrows=0)  # just header
            # get row count quickly
            with open(dataset_path, "r", encoding="utf-8") as f:
                rows = sum(1 for _ in f) - 1  # header
            info["data"]["dataset"] = {"exists": True, "rows": rows, "columns": list(df.columns)}
            # sample dates if possible
            try:
                df2 = pd.read_csv(dataset_path, usecols=["event_date"])
                if not df2.empty:
                    info["data"]["dataset"]["first_date"] = str(df2["event_date"].min())
                    info["data"]["dataset"]["last_date"] = str(df2["event_date"].max())
            except Exception:
                pass
        except Exception as e:
            info["data"]["dataset"] = {"exists": True, "error": str(e)}
    else:
        info["data"]["dataset"] = {"exists": False}

    # Model
    if model_path.exists():
        info["model"]["exists"] = True
        info["model"]["size_bytes"] = model_path.stat().st_size
        info["model"]["mtime"] = datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    else:
        info["model"]["exists"] = False

    if features_path.exists():
        info["model"]["features_exists"] = True
        info["model"]["features_size_bytes"] = features_path.stat().st_size
        info["model"]["features_mtime"] = datetime.fromtimestamp(features_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        try:
            meta = joblib.load(features_path)
            info["model"]["meta"] = {
                "model_type": meta.get("model_type"),
                "feature_cols": len(meta.get("feature_cols_final", [])),
                "raw_feature_cols": meta.get("raw_feature_cols"),
                "test_roc_auc": meta.get("test_roc_auc"),
                "test_accuracy": meta.get("test_accuracy"),
                "calibration": meta.get("calibration"),
                "composite_method": meta.get("composite_method"),
            }
            # also show training date from calibration?
        except Exception as e:
            info["model"]["meta_error"] = str(e)
    else:
        info["model"]["features_exists"] = False

    # Plot
    plot_path = Path(BASE_DIR) / "models" / "stacking_ensemble_feature_importance.png"
    if plot_path.exists():
        info["model"]["plot_exists"] = True
        info["model"]["plot_mtime"] = datetime.fromtimestamp(plot_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    else:
        info["model"]["plot_exists"] = False

    if as_json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    # Human table
    print()
    print("=" * 70)
    print(f"  UFC INFO")
    print("=" * 70)
    print(f"  Cutoff date: {info['cutoff_date']}")
    print()
    print("  Data:")
    for key in ["events_index", "fights", "fighters_cache", "dataset"]:
        d = info["data"].get(key, {})
        exists = d.get("exists")
        if not exists:
            print(f"    {key:<20} not found ({info['paths'][key]})")
            continue
        if "error" in d:
            print(f"    {key:<20} error: {d['error']}")
            continue
        line = f"    {key:<20} {d.get('count', d.get('rows', '?'))} "
        if key == "dataset":
            line = f"    {key:<20} {d.get('rows', '?')} rows, {len(d.get('columns', []))} cols"
        if "first_date" in d:
            line += f"  [{d['first_date']} -> {d['last_date']}]"
        if key == "events_index" and "scrapped" in d:
            line += f"  scrapped={d['scrapped']} pending={d['pending']}"
        print(line)
        if key == "dataset" and "columns" in d:
            # show truncation
            cols = d["columns"][:6]
            print(f"      columns: {', '.join(cols)}{'...' if len(d['columns'])>6 else ''}")

    print()
    print("  Model:")
    for k in ["exists", "features_exists", "plot_exists"]:
        # map to path keys
        pass
    m = info["model"]
    if not m.get("exists"):
        print(f"    model: not found ({info['paths']['model']})")
    else:
        print(f"    model: {info['paths']['model']}  ({m['size_bytes']} bytes, {m['mtime']})")
    if not m.get("features_exists"):
        print(f"    features: not found ({info['paths']['features']})")
    else:
        print(f"    features: {info['paths']['features']}  ({m['features_size_bytes']} bytes, {m['features_mtime']})")
        meta = m.get("meta")
        if meta:
            print(f"      type: {meta.get('model_type')}  features={meta.get('feature_cols')}  roc_auc={meta.get('test_roc_auc')}  acc={meta.get('test_accuracy')}")
            cal = meta.get("calibration")
            if cal:
                print(f"      calibration: {cal.get('method')}  n_oof={cal.get('n_oof')}  ll={cal.get('cal_test_log_loss'):.4f}" if cal.get("cal_test_log_loss") else f"      calibration: {cal}")
    if m.get("plot_exists"):
        print(f"    plot: {BASE_DIR / 'models' / 'stacking_ensemble_feature_importance.png'}  ({m['plot_mtime']})")
    else:
        print(f"    plot: not found")
    print()
    print("  Paths:")
    for k, p in info["paths"].items():
        print(f"    {k:<15} {p}")
    print()
    return 0
