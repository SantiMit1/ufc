"""
Batch predict multiple UFC fights.

Usage:
  .venv/Scripts/python src/predict_batch.py "FighterA,FighterB,WeightClass" "FighterC,FighterD,WeightClass"
  .venv/Scripts/python src/predict_batch.py "FighterA,FighterB"

Each arg: Fighter1,Fighter2[,WeightClass]. WeightClass defaults to "Catch Weight".
"""
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import (
    make_initial_state, compute_stats_from_state, build_fighter_states,
    safe_sub, constrain_round_probas,
)
from ensemble_utils import ChronologicalStackingEnsemble, ChronologicalStackingEnsembleMultiClass
from stats_utils import _prior_accum_init, _prior_accum_add, _get_current_priors


BASE_DIR = Path(__file__).resolve().parent.parent
FIGHTS_PATH = BASE_DIR / "data" / "fights.json"
FIGHTERS_CACHE_PATH = BASE_DIR / "data" / "fighters_cache.json"
MODEL_PATH = BASE_DIR / "models" / "ufc_stacking_ensemble.pkl"
FEATURE_COLS_PATH = BASE_DIR / "models" / "ufc_stacking_ensemble_meta.pkl"
METHOD_MODEL_PATH = BASE_DIR / "models" / "ufc_method_model.pkl"
ROUND_MODEL_PATH = BASE_DIR / "models" / "ufc_round_model.pkl"

CUTOFF_DATE = datetime(2001, 1, 1)

WEIGHT_CLASSES = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight", "Catch Weight",
]


def main():
    parser = argparse.ArgumentParser(description="Batch predict UFC fights")
    parser.add_argument("fights", nargs="+", help="Fighter1,Fighter2[,WeightClass[,Rounds]]")
    args = parser.parse_args()

    parsed = []
    for arg in args.fights:
        parts = [p.strip() for p in arg.split(",")]
        if len(parts) == 2:
            f1, f2 = parts
            cat = "Catch Weight"
            rounds = 3
        elif len(parts) == 3:
            f1, f2, third = parts
            if third in ("3", "5"):
                cat = "Catch Weight"
                rounds = int(third)
            else:
                cat = third
                if cat not in WEIGHT_CLASSES:
                    print(f"Warning: '{cat}' is not a standard weight class, using as-is.")
                rounds = 3
        elif len(parts) == 4:
            f1, f2, cat, r_str = parts
            rounds = int(r_str) if r_str in ("3", "5") else 3
        else:
            print(f"Invalid format: '{arg}'. Use 'Fighter1,Fighter2[,WeightClass[,Rounds]]'.")
            sys.exit(1)
        parsed.append((f1, f2, cat, rounds))

    with open(FIGHTS_PATH) as f:
        fights = json.load(f)
    with open(FIGHTERS_CACHE_PATH) as f:
        fighters_cache = json.load(f)

    _prior_accum_init()
    for fight in fights:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    for fight in sorted(fights, key=lambda f: f["_parsed_date"]):
        if fight["_parsed_date"] >= CUTOFF_DATE:
            _prior_accum_add(fight)
    priors = _get_current_priors()

    model = joblib.load(MODEL_PATH)
    feature_meta = joblib.load(FEATURE_COLS_PATH)

    method_model = None
    round_model = None
    try:
        method_model = joblib.load(METHOD_MODEL_PATH)
    except Exception:
        pass
    try:
        round_model = joblib.load(ROUND_MODEL_PATH)
    except Exception:
        pass

    fighter_states = build_fighter_states(fights, fighters_cache)
    current_date = datetime.now()

    def get_phys(name, key):
        v = fighters_cache.get(name, {}).get(key)
        return float(v) if v is not None else np.nan

    def predict_fight(f1, f2, category, max_rounds=3):
        def predict_order(a, b):
            height1, reach1 = get_phys(a, "height_cm"), get_phys(a, "reach_cm")
            height2, reach2 = get_phys(b, "height_cm"), get_phys(b, "reach_cm")

            state1 = fighter_states.get(a, make_initial_state())
            state2 = fighter_states.get(b, make_initial_state())

            feat1 = compute_stats_from_state(state1, a, fighters_cache, current_date, category=category, priors=priors)
            feat2 = compute_stats_from_state(state2, b, fighters_cache, current_date, category=category, priors=priors)

            row = {}
            diff_fields = [
                "win_pct", "ko_rate", "sub_rate", "dec_rate",
                "ko_loss_rate", "sub_loss_rate",
                "sig_str_landed_per_min", "sig_str_absorbed_per_min", "sig_str_accuracy",
                "td_avg_per_15min", "td_accuracy", "td_defense",
                "sub_att_per_15min", "ctrl_time_pct", "days_since_last_fight",
            ]
            row["age_a"] = feat1["age"]
            row["age_b"] = feat2["age"]
            row["stance_a"] = feat1["stance"]
            row["stance_b"] = feat2["stance"]
            row["category"] = category
            row["age_diff"] = safe_sub(feat1["age"], feat2["age"])
            row["height_diff"] = safe_sub(height1, height2)
            row["reach_diff"] = safe_sub(reach1, reach2)
            row["win_streak_diff"] = feat1["current_win_streak"] - feat2["current_win_streak"]
            row["losing_streak_diff"] = feat1["current_losing_streak"] - feat2["current_losing_streak"]
            row["total_fights_diff"] = feat1["total_fights"] - feat2["total_fights"]
            row["elo_diff"] = feat1["elo"] - feat2["elo"]

            for field in diff_fields:
                row[f"{field}_diff"] = safe_sub(feat1[field], feat2[field])

            new_fighter_fields = [
                "recent_3_wins", "recent_3_losses", "recent_5_wins", "recent_5_losses",
                "recent_3_ko_loss_rate", "recent_5_ko_loss_rate",
                "decay_sig_per_min", "decay_sig_absorbed_per_min", "decay_td_per_15min",
                "avg_opp_elo", "avg_opp_elo_wins",
            ]
            for field in new_fighter_fields:
                row[f"{field}_diff"] = safe_sub(feat1[field], feat2[field])

            raw_cols = feature_meta["raw_feature_cols"]
            X_raw = pd.DataFrame([row])[raw_cols]
            for c in feature_meta["numeric_cols"]:
                if c in X_raw.columns:
                    X_raw[c] = X_raw[c].astype(float)
            for c in feature_meta["numeric_cols"]:
                if c in X_raw.columns and c in feature_meta.get("medians", {}):
                    X_raw[c] = X_raw[c].fillna(feature_meta["medians"][c])
            X_encoded = pd.get_dummies(X_raw, columns=feature_meta["cat_cols"], drop_first=True)
            for col in feature_meta["feature_cols_final"]:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0
            X_encoded = X_encoded[feature_meta["feature_cols_final"]]
            prob = model.predict_proba(X_encoded)[0, 1]

            method_proba = None
            round_proba = None
            if method_model is not None:
                method_proba = method_model.predict_proba(X_encoded)[0]
            if round_model is not None:
                round_proba = round_model.predict_proba(X_encoded)[0]

            return prob, method_proba, round_proba

        prob_a_forward, method_a_forward, round_a_forward = predict_order(f1, f2)
        prob_b_forward, method_b_forward, round_b_forward = predict_order(f2, f1)
        prob_a = (prob_a_forward + (1.0 - prob_b_forward)) / 2.0
        prob_b = 1.0 - prob_a

        method_proba = None
        round_proba = None
        if method_a_forward is not None and method_b_forward is not None:
            method_proba = (method_a_forward + method_b_forward) / 2.0
        if round_a_forward is not None and round_b_forward is not None:
            round_proba = (round_a_forward + round_b_forward) / 2.0
            round_proba = constrain_round_probas(round_proba, max_rounds)

        return prob_a, prob_b, method_proba, round_proba

    results = []
    skipped = []
    warned = []

    for f1, f2, cat, rounds in parsed:
        state1 = fighter_states.get(f1, make_initial_state())
        state2 = fighter_states.get(f2, make_initial_state())
        tf1 = state1["total_fights"]
        tf2 = state2["total_fights"]

        if tf1 == 0 or tf2 == 0:
            skipped.append((f1, f2, tf1, tf2))
            continue

        if tf1 < 3 or tf2 < 3:
            warned.append((f1, f2, tf1, tf2))

        prob_a, prob_b, method_proba, round_proba = predict_fight(f1, f2, cat, rounds)
        results.append((f1, f2, cat, prob_a, prob_b, tf1, tf2, method_proba, round_proba, rounds))

    # Output
    print()
    print("=" * 130)
    print("  BATCH PREDICTIONS")
    print("=" * 130)
    method_available = any(r[7] is not None for r in results)
    if method_available:
        header = f"  {'#':<4s} {'Fighter A':<26s} {'Fighter B':<26s} {'Category':<18s} {'Prob A':<7s} {'Prob B':<7s} {'Method':<14s} {'Round':<13s} {'MaxR':<4s}"
    else:
        header = f"  {'#':<4s} {'Fighter A':<28s} {'Fighter B':<28s} {'Category':<22s} {'Prob A':<8s} {'Prob B':<8s} {'A fights':<9s} {'B fights':<9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    warned_set = {(f1, f2) for f1, f2, _, _ in warned}

    for i, (f1, f2, cat, prob_a, prob_b, tf1, tf2, method_proba, round_proba, rounds) in enumerate(results, 1):
        warn_flag = "  *" if (f1, f2) in warned_set else ""
        if method_available and method_proba is not None and round_proba is not None:
            method_labels = ["KO", "SUB", "DEC"]
            best_method_idx = int(method_proba.argmax())
            best_method = method_labels[best_method_idx]
            method_str = f"{best_method} ({method_proba[best_method_idx]*100:.1f}%)"
            if best_method_idx == 2:  # DEC
                round_str = "-"
            else:
                best_round = int(round_proba.argmax()) + 1
                round_str = f"R{best_round} ({round_proba[best_round-1]*100:.1f}%)"
            print(f"  {i:<4d} {f1:<26s} {f2:<26s} {cat:<18s} {prob_a*100:<6.1f}% {prob_b*100:<6.1f}% {method_str:<14s} {round_str:<13s} {rounds}{warn_flag}")
        else:
            print(f"  {i:<4d} {f1:<28s} {f2:<28s} {cat:<22s} {prob_a*100:<7.1f}% {prob_b*100:<7.1f}% {tf1:<9d} {tf2:<9d}{warn_flag}")

    print("=" * 120)

    if warned:
        print("\n  * WARNING: One or both fighters have fewer than 3 UFC fights — prediction may be unreliable.")
        for f1, f2, tf1, tf2 in warned:
            print(f"    - {f1} ({tf1} fights) vs {f2} ({tf2} fights)")

    if skipped:
        print(f"\n  SKIPPED ({len(skipped)} fight(s) — no fight history for one or both fighters):")
        for f1, f2, tf1, tf2 in skipped:
            print(f"    - {f1} (0 fights) vs {f2} (0 fights)" if tf1 == 0 and tf2 == 0 else
                  f"    - {f1} ({tf1} fights) vs {f2} ({tf2} fights)")

    print()


if __name__ == "__main__":
    import argparse
    main()
