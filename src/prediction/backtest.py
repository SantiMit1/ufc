"""
Backtest the stacked ensemble on a historical period without look-ahead.

Usage:
  .venv/Scripts/python src/prediction/backtest.py --start 2015-01-01
  .venv/Scripts/python src/prediction/backtest.py --start 2015-01-01 --end 2019-12-31

Fights are processed chronologically. For every fight in [start, end] the
model only sees states/priors built from fights BEFORE that one (no
look-ahead). Fights where a fighter is debuting are skipped: a fighter's
debut fight is detected via their ``debut_date`` in the fighters cache
(fallback: 0 prior fights in the state), so a fighter needs at least 1 prior
fight for the fight to count in the evaluation. Fights without a winner
(draw / no contest) are skipped too.
"""
import argparse
import json
import random
import sys
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    FIGHTS_PATH, FIGHTERS_CACHE_PATH, MODEL_PATH, FEATURE_COLS_PATH, CUTOFF_DATE,
)
from fighter_engine import (
    make_initial_state, predict_fight, classify_method, get_k_factor,
    apply_elo_decay, elo_update, update_state,
)
from stats_utils import _prior_accum_init, _prior_accum_add, _get_current_priors
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from tqdm import tqdm


def is_debut_fighter(fighter_name, event_date, fighters_cache, fighter_state):
    """True if this fight is the fighter's debut (0 prior fights)."""
    entry = fighters_cache.get(fighter_name, {})
    debut = entry.get("debut_date")
    if debut:
        return debut == event_date
    return fighter_state.get(fighter_name, make_initial_state())["total_fights"] == 0


def main():
    parser = argparse.ArgumentParser(description="Backtest the model without look-ahead")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), inclusive (default: today)")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--features-path", default=str(FEATURE_COLS_PATH))
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    if end < start:
        parser.error("--end must be >= --start")

    with open(FIGHTS_PATH, encoding="utf-8") as f:
        fights = json.load(f)
    with open(FIGHTERS_CACHE_PATH, encoding="utf-8") as f:
        fighters_cache = json.load(f)

    model = joblib.load(args.model_path)
    feature_meta = joblib.load(args.features_path)

    random.seed(42)

    _prior_accum_init()
    fighter_state = {}

    for fight in fights:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    filtered = sorted(
        (f for f in fights if f["_parsed_date"] >= CUTOFF_DATE),
        key=lambda f: f["_parsed_date"],
    )

    total = len(filtered)
    in_period = 0
    skipped_debut = 0
    skipped_no_winner = 0
    skipped_error = 0
    probs = []

    period_fights = sum(1 for f in filtered if start <= f["_parsed_date"] <= end)
    pbar = tqdm(total=period_fights, unit="fight", desc="Backtest", ncols=100)

    for fight in filtered:
        f1, f2 = fight["fighter_1"], fight["fighter_2"]
        if f1 not in fighter_state:
            fighter_state[f1] = make_initial_state()
        if f2 not in fighter_state:
            fighter_state[f2] = make_initial_state()

        fighter_state[f1]["elo"] = apply_elo_decay(
            fighter_state[f1]["elo"], fighter_state[f1]["last_fight_date"], fight["_parsed_date"])
        fighter_state[f2]["elo"] = apply_elo_decay(
            fighter_state[f2]["elo"], fighter_state[f2]["last_fight_date"], fight["_parsed_date"])

        priors = _get_current_priors()

        if start <= fight["_parsed_date"] <= end:
            in_period += 1
            if is_debut_fighter(f1, fight["event_date"], fighters_cache, fighter_state) or \
               is_debut_fighter(f2, fight["event_date"], fighters_cache, fighter_state):
                skipped_debut += 1
            elif fight["winner"] not in (f1, f2):
                skipped_no_winner += 1
            else:
                try:
                    # Randomly assign fighter_a/fighter_b sides (like the dataset
                    # build) so winner labels are balanced; the model averages
                    # both orderings so predictions are order-independent.
                    if random.random() < 0.5:
                        a_name, b_name = f1, f2
                    else:
                        a_name, b_name = f2, f1
                    prob_a, _ = predict_fight(
                        a_name, b_name, fight["category"], fighter_state, fighters_cache,
                        model, feature_meta, fight["_parsed_date"], priors=priors)
                    actual = 1 if fight["winner"] == a_name else 0
                    probs.append((prob_a, actual, fight["event_date"]))
                except Exception as e:
                    skipped_error += 1
                    print(f"    [WARN] prediction failed for {f1} vs {f2} ({fight['event_date']}): {e}")

            pbar.update(1)
            pbar.set_postfix_str(f"eval={len(probs)} debut={skipped_debut}")

        _prior_accum_add(fight)

        is_win_loss, win_side, finish_type = classify_method(
            fight["method"], fight["winner"], f1, f2)
        f1_elo_before = fighter_state[f1]["elo"]
        f2_elo_before = fighter_state[f2]["elo"]
        update_state(fighter_state[f1], fight, True, is_win_loss, win_side,
                     finish_type, opponent_elo=f2_elo_before)
        update_state(fighter_state[f2], fight, False, is_win_loss, win_side,
                     finish_type, opponent_elo=f1_elo_before)
        if is_win_loss:
            score_a = 1.0 if win_side == 1 else 0.0
            k_a = get_k_factor(fighter_state[f1]["total_fights"])
            k_b = get_k_factor(fighter_state[f2]["total_fights"])
            fighter_state[f1]["elo"], fighter_state[f2]["elo"] = elo_update(
                fighter_state[f1]["elo"], fighter_state[f2]["elo"], score_a,
                k_a=k_a, k_b=k_b)

    pbar.close()

    # ── Metrics ────────────────────────────────────────────────────────────────
    n_eval = len(probs)
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
        return

    arr = np.array(probs, dtype=object)
    y_prob = np.array([float(r[0]) for r in probs])
    y_true = np.array([int(r[1]) for r in probs])

    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    ll = log_loss(y_true, y_prob, labels=[0, 1])
    bs = brier_score_loss(y_true, y_prob)
    if len(set(y_true)) > 1:
        roc = roc_auc_score(y_true, y_prob)
    else:
        roc = float("nan")

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


if __name__ == "__main__":
    main()
