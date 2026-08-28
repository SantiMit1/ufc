"""Backtest command."""

import json
import json as _json
import random
import sys
from datetime import datetime

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm

from cli._common import resolve_paths
from fighter_engine import FightStateEngine, is_debut, predict_fight
from stats_utils import PriorAccumulator, load_fighter_cache, load_fights


def handle_backtest(args):
    fights_path, cache_path, _, model_path, features_path = resolve_paths(args)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    if end < start:
        print("error: --end must be >= --start", file=sys.stderr)
        return 2

    as_json = args.json


    # Load with custom paths if --data-dir supplied
    if str(fights_path) != str(__import__("config").FIGHTS_PATH) or str(cache_path) != str(__import__("config").FIGHTERS_CACHE_PATH):
        with open(fights_path, "r", encoding="utf-8") as f:
            fights = _json.load(f)
        with open(cache_path, "r", encoding="utf-8") as f:
            fighters_cache = _json.load(f)
    else:
        fights = load_fights()
        fighters_cache = load_fighter_cache()

    model = joblib.load(model_path)
    feature_meta = joblib.load(features_path)

    random.seed(42)
    prior_accum = PriorAccumulator()
    engine = FightStateEngine(fights)
    fighter_state = engine.state
    filtered = engine.filtered
    total = len(filtered)
    in_period = 0
    skipped_debut = 0
    skipped_no_winner = 0
    skipped_error = 0
    probs = []

    period_fights = sum(1 for f in filtered if start <= f["_parsed_date"] <= end)
    # tqdm writes to stderr; suppress if json?
    show_pbar = not as_json
    pbar = tqdm(total=period_fights, unit="fight", desc="Backtest", ncols=100, disable=not show_pbar)

    for fight in engine:
        f1, f2 = fight["fighter_1"], fight["fighter_2"]
        priors = prior_accum.priors()
        if start <= fight["_parsed_date"] <= end:
            in_period += 1
            if is_debut(f1, fighter_state, fighters_cache, fight["event_date"]) or is_debut(f2, fighter_state, fighters_cache, fight["event_date"]):
                skipped_debut += 1
            elif fight["winner"] not in (f1, f2):
                skipped_no_winner += 1
            else:
                try:
                    if random.random() < 0.5:
                        a_name, b_name = f1, f2
                    else:
                        a_name, b_name = f2, f1
                    prob_a, _ = predict_fight(a_name, b_name, fight["category"], fighter_state, fighters_cache, model, feature_meta, fight["_parsed_date"], priors=priors)
                    actual = 1 if fight["winner"] == a_name else 0
                    probs.append((prob_a, actual, fight["event_date"]))
                except Exception as e:
                    skipped_error += 1
                    if not as_json:
                        print(f"    [WARN] prediction failed for {f1} vs {f2} ({fight['event_date']}): {e}")
            if show_pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"eval={len(probs)} debut={skipped_debut}")
            elif as_json:
                # still update but not display
                pass
        prior_accum.add(fight)

    if show_pbar:
        pbar.close()

    n_eval = len(probs)
    if as_json:
        if n_eval == 0:
            out = {"start": args.start, "end": args.end or datetime.now().strftime("%Y-%m-%d"), "total": total, "in_period": in_period, "evaluated": 0, "skipped": {"debut": skipped_debut, "no_winner": skipped_no_winner, "error": skipped_error}, "metrics": None}
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0
        y_prob = np.array([float(r[0]) for r in probs])
        y_true = np.array([int(r[1]) for r in probs])
        acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
        ll = log_loss(y_true, y_prob, labels=[0, 1])
        bs = brier_score_loss(y_true, y_prob)
        roc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
        # calibration bins
        bins = np.linspace(0.0, 1.0, 11)
        bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, 9)
        calib = []
        for i in range(10):
            mask = bin_indices == i
            nb = int(mask.sum())
            if nb == 0:
                continue
            calib.append({"bin": f"[{bins[i]:.1f}-{bins[i+1]:.1f})", "n": nb, "avg_prob": float(y_prob[mask].mean()), "obs_freq": float(y_true[mask].mean())})
        years = sorted(set(r[2][:4] for r in probs))
        per_year = []
        for year in years:
            mask = np.array([r[2][:4] == year for r in probs])
            yp = y_prob[mask]
            yt = y_true[mask]
            a = accuracy_score(yt, (yp >= 0.5).astype(int))
            r_ = roc_auc_score(yt, yp) if len(set(yt)) > 1 else float("nan")
            per_year.append({"year": year, "n": int(mask.sum()), "acc": float(a), "auc": float(r_) if not np.isnan(r_) else None, "avg_prob": float(yp.mean())})
        out = {
            "start": args.start,
            "end": args.end or datetime.now().strftime("%Y-%m-%d"),
            "total": total,
            "in_period": in_period,
            "evaluated": n_eval,
            "skipped": {"debut": skipped_debut, "no_winner": skipped_no_winner, "error": skipped_error},
            "metrics": {"accuracy": float(acc), "log_loss": float(ll), "brier": float(bs), "roc_auc": float(roc) if not np.isnan(roc) else None},
            "calibration": calib,
            "per_year": per_year,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # Table output (default)
    print("\n" + "=" * 70)
    print(f"  BACKTEST  {start.date()}  ->  {end.date()}")
    print("=" * 70)
    print(f"  Total fights processed:      {total}")
    print(f"  Fights in period:            {in_period}")
    print(f"  Evaluated:                   {n_eval}")
    print(f"  Skipped (debut fighter):     {skipped_debut}")
    print(f"  Skipped (no winner):         {skipped_no_winner}")
    if skipped_error:
        print(f"  Skipped (prediction error):  {skipped_error}")
    if n_eval == 0:
        print("\n  No evaluable fights in the requested period.")
        return 0
    y_prob = np.array([float(r[0]) for r in probs])
    y_true = np.array([int(r[1]) for r in probs])
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    ll = log_loss(y_true, y_prob, labels=[0, 1])
    bs = brier_score_loss(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    print(f"\n  Accuracy:    {acc:.4f}")
    print(f"  Log Loss:    {ll:.4f}")
    print(f"  Brier score: {bs:.4f}")
    print(f"  ROC-AUC:     {roc:.4f}")
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, 9)
    print(f"\n  {'Calibration':^50s}")
    print(f"  {'Bin range':<14s} {'n':>5s} {'avg_prob':>9s} {'obs_freq':>9s}")
    print("  " + "-" * 37)
    for i in range(10):
        mask = bin_indices == i
        nb = int(mask.sum())
        if nb == 0:
            continue
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f})  {nb:5d} {y_prob[mask].mean():9.3f} {y_true[mask].mean():9.3f}")
    years = sorted(set(r[2][:4] for r in probs))
    if len(years) > 1:
        print(f"\n  {'Year':<6s} {'n':>5s} {'acc':>7s} {'auc':>7s} {'avg_prob':>9s}")
        print("  " + "-" * 34)
        for year in years:
            mask = np.array([r[2][:4] == year for r in probs])
            yp = y_prob[mask]
            yt = y_true[mask]
            a = accuracy_score(yt, (yp >= 0.5).astype(int))
            r_ = roc_auc_score(yt, yp) if len(set(yt)) > 1 else float("nan")
            print(f"  {year:<6s} {mask.sum():5d} {a:7.3f} {r_:7.3f} {yp.mean():9.3f}")
    print()
    return 0
