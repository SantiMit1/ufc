"""
Evaluate value betting performance on past UFC events.

Usage:
  1. Edit the EVALUATION_FIGHTS list below with past fights, odds, and winners.
  2. Run: .venv/Scripts/python evaluate_valuebets.py

No data leakage: each event is predicted using ONLY fights that took place
before that event's date.
"""
import copy
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from predict import (
    make_initial_state, compute_stats_from_state, safe_sub,
    classify_method, get_k_factor, apply_elo_decay, elo_update,
    update_state, ELO_INITIAL,
)
from ensemble_utils import ChronologicalStackingEnsemble
from stats_utils import _prior_accum_init, _prior_accum_add, _get_current_priors

BASE_DIR = Path(__file__).resolve().parent
FIGHTS_PATH = BASE_DIR / "data" / "fights.json"
FIGHTERS_CACHE_PATH = BASE_DIR / "data" / "fighters_cache.json"
MODEL_PATH = BASE_DIR / "models" / "ufc_stacking_ensemble.pkl"
FEATURE_COLS_PATH = BASE_DIR / "models" / "ufc_stacking_ensemble_meta.pkl"

CUTOFF_DATE = datetime(2001, 1, 1)
ELO_K = 96


# -----------------------------------------------------------------------------
#  PAST FIGHTS TO EVALUATE  (edit this list)
# -----------------------------------------------------------------------------
EVALUATION_FIGHTS = [
    # {
    #     "fighter_a": "Alessandro Costa",
    #     "fighter_b": "Cody Durden",
    #     "odds_a": 1.39,
    #     "odds_b": 3.03,
    #     "category": "Flyweight",
    #     "winner": "Cody Durden",
    #     "event_date": "2024-12-07",
    # },
]


def _build_fighter_states_subset(fights: list, fighters_cache: dict) -> dict:
    """Same as predict.build_fighter_states but without modifying originals."""
    fights_copy = copy.deepcopy(fights)
    for fight in fights_copy:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    filtered = sorted(
        [f for f in fights_copy if f["_parsed_date"] >= CUTOFF_DATE],
        key=lambda f: f["_parsed_date"],
    )
    fighter_state: dict[str, dict] = {}
    for fight in filtered:
        f1, f2 = fight["fighter_1"], fight["fighter_2"]
        if f1 not in fighter_state:
            fighter_state[f1] = make_initial_state()
        if f2 not in fighter_state:
            fighter_state[f2] = make_initial_state()
        fighter_state[f1]["elo"] = apply_elo_decay(
            fighter_state[f1]["elo"], fighter_state[f1]["last_fight_date"], fight["_parsed_date"]
        )
        fighter_state[f2]["elo"] = apply_elo_decay(
            fighter_state[f2]["elo"], fighter_state[f2]["last_fight_date"], fight["_parsed_date"]
        )
        is_win_loss, win_side, finish_type = classify_method(
            fight["method"], fight["winner"], f1, f2
        )
        f1_elo_before = fighter_state[f1]["elo"]
        f2_elo_before = fighter_state[f2]["elo"]
        update_state(fighter_state[f1], fight, True, is_win_loss, win_side, finish_type, opponent_elo=f2_elo_before)
        update_state(fighter_state[f2], fight, False, is_win_loss, win_side, finish_type, opponent_elo=f1_elo_before)
        if is_win_loss:
            score_a = 1.0 if win_side == 1 else 0.0
            k_a = get_k_factor(fighter_state[f1]["total_fights"])
            k_b = get_k_factor(fighter_state[f2]["total_fights"])
            f1_new, f2_new = elo_update(fighter_state[f1]["elo"], fighter_state[f2]["elo"], score_a, k_a=k_a, k_b=k_b)
            fighter_state[f1]["elo"] = f1_new
            fighter_state[f2]["elo"] = f2_new
    return fighter_state


def evaluate_valuebets(event_fights: list[dict], bankroll: float = 4000.0) -> None:
    """
    Evaluate value betting performance on past events.

    Each fight dict must include:
        fighter_a, fighter_b, odds_a, odds_b, category, winner, event_date

    No lookahead: for each unique event_date, builds fighter states using
    ONLY fights from data/fights.json that took place BEFORE that date.
    """
    with open(FIGHTS_PATH, encoding="utf-8") as f:
        all_fights = json.load(f)
    with open(FIGHTERS_CACHE_PATH, encoding="utf-8") as f:
        fighters_cache = json.load(f)

    model = joblib.load(MODEL_PATH)
    feature_meta = joblib.load(FEATURE_COLS_PATH)

    for f in all_fights:
        f["_parsed_date"] = datetime.strptime(f["event_date"], "%Y-%m-%d")

    sorted_fights = sorted(event_fights, key=lambda x: x["event_date"])
    by_date = OrderedDict()
    for f in sorted_fights:
        by_date.setdefault(f["event_date"], []).append(f)

    all_results = []
    all_valuebets = []
    all_skipped = []
    all_warned = []

    for event_date, fights_on_date in by_date.items():
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        historical = [f for f in all_fights if f["_parsed_date"] < event_dt]

        _prior_accum_init()
        for f in sorted(historical, key=lambda x: x["_parsed_date"]):
            _prior_accum_add(f)
        priors = _get_current_priors()

        fighter_states = _build_fighter_states_subset(historical, fighters_cache)
        current_date = event_dt

        for fight in fights_on_date:
            f1 = fight["fighter_a"]
            f2 = fight["fighter_b"]
            cat = fight.get("category", "Catch Weight")
            odds_a = fight["odds_a"]
            odds_b = fight["odds_b"]
            winner = fight.get("winner", "")

            state1 = fighter_states.get(f1, make_initial_state())
            state2 = fighter_states.get(f2, make_initial_state())

            if state1["total_fights"] == 0 or state2["total_fights"] == 0:
                all_skipped.append((f1, f2, state1["total_fights"], state2["total_fights"]))
                continue

            if state1["total_fights"] < 3 or state2["total_fights"] < 3:
                all_warned.append((f1, f2, state1["total_fights"], state2["total_fights"]))

            def get_phys(name, key):
                v = fighters_cache.get(name, {}).get(key)
                return float(v) if v is not None else np.nan

            def predict_order(a, b):
                height1, reach1 = get_phys(a, "height_cm"), get_phys(a, "reach_cm")
                height2, reach2 = get_phys(b, "height_cm"), get_phys(b, "reach_cm")

                s1 = fighter_states.get(a, make_initial_state())
                s2 = fighter_states.get(b, make_initial_state())

                feat1 = compute_stats_from_state(s1, a, fighters_cache, current_date, category=cat, priors=priors)
                feat2 = compute_stats_from_state(s2, b, fighters_cache, current_date, category=cat, priors=priors)

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
                row["category"] = cat
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
                return model.predict_proba(X_encoded)[0, 1]

            prob_a_forward = predict_order(f1, f2)
            prob_b_forward = predict_order(f2, f1)
            prob_a = (prob_a_forward + (1.0 - prob_b_forward)) / 2.0
            prob_b = 1.0 - prob_a

            fair_a = 1.0 / prob_a if prob_a > 0 else float("inf")
            fair_b = 1.0 / prob_b if prob_b > 0 else float("inf")

            implied_a = 1.0 / odds_a
            implied_b = 1.0 / odds_b
            total_implied = implied_a + implied_b
            prob_market_a = implied_a / total_implied
            prob_market_b = implied_b / total_implied
            edge_a = (prob_a / prob_market_a) - 1.0
            edge_b = (prob_b / prob_market_b) - 1.0

            model_pick = f1 if prob_a >= prob_b else f2
            model_correct = (model_pick == winner)

            value_bet_info = None
            for pick_fighter, prob_v, odds_v, market_p, edge_v in [
                (f1, prob_a, odds_a, prob_market_a, edge_a),
                (f2, prob_b, odds_b, prob_market_b, edge_b),
            ]:
                if edge_v >= 0.15:
                    stake = bankroll * 0.25 * edge_v / (odds_v - 1.0)
                    won = (pick_fighter == winner)
                    value_bet_info = {
                        "fight": f"{f1} vs {f2}",
                        "pick": pick_fighter,
                        "model_prob": prob_v,
                        "market_prob": market_p,
                        "odds": odds_v,
                        "edge": edge_v,
                        "stake": stake,
                        "won": won,
                    }
                    all_valuebets.append(value_bet_info)
                    break

            all_results.append({
                "fighter_a": f1, "fighter_b": f2, "category": cat,
                "event_date": event_date, "winner": winner,
                "prob_a": prob_a, "prob_b": prob_b,
                "fair_odds_a": fair_a, "fair_odds_b": fair_b,
                "odds_a": odds_a, "odds_b": odds_b,
                "model_pick": model_pick, "model_correct": model_correct,
                "value_bet": value_bet_info,
            })

    # ── Output: same format as bets.py ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FIGHT PREDICTIONS")
    print("=" * 80)
    print(f"  {'Pe #':<6s} {'Fighter A':<28s} {'Fighter B':<28s} {'Prob A':<8s} {'Prob B':<8s} {'Fair Odds A':<11s} {'Fair Odds B':<11s}")
    print("  " + "-" * 100)

    for i, r in enumerate(all_results, 1):
        print(f"  {i:<6d} {r['fighter_a']:<28s} {r['fighter_b']:<28s} {r['prob_a']*100:<8.1f} {r['prob_b']*100:<8.1f} {r['fair_odds_a']:<11.2f} {r['fair_odds_b']:<11.2f}")

    print()
    print("=" * 80)
    print("  VALUE BETS  (edge >= 15%)")
    print("=" * 80)

    if not all_valuebets:
        print("\n  No value bets found with edge >= 15%.\n")
    else:
        print(f"  Bankroll: ${bankroll:.2f}  |  Quarter Kelly")
        print()
        print(f"  {'Fight':<35s} {'Pick':<28s} {'Model Prob':<11s} {'Market Prob':<12s} {'Odds':<7s} {'Edge%':<8s}  {'Bank%':<7s} {'Stake $':<10s}")
        print("  " + "-" * 119)
        for vb in all_valuebets:
            bank_pct = vb["stake"] / bankroll * 100
            print(f"  {vb['fight']:<35s} {vb['pick']:<28s} {vb['model_prob']*100:<10.1f}% {vb['market_prob']*100:<10.1f}% {vb['odds']:<7.2f} {vb['edge']*100:<7.1f}%  {bank_pct:<6.2f}% ${vb['stake']:<8.2f}")
        print("  " + "-" * 111)
        total_stake = sum(vb["stake"] for vb in all_valuebets)
        print(f"\n  Total stake across all value bets: ${total_stake:.2f}")
        print(f"  Remaining bankroll: ${bankroll - total_stake:.2f}\n")

    # ── Warnings ──
    if all_warned:
        print("=" * 80)
        print("  WARNINGS (fighters with < 3 UFC fights)")
        print("=" * 80)
        for f1, f2, tf1, tf2 in all_warned:
            print(f"  - {f1} ({tf1} fights) vs {f2} ({tf2} fights)")
        print()

    if all_skipped:
        print("=" * 80)
        print("  SKIPPED FIGHTS (fighters with 0 UFC fights)")
        print("=" * 80)
        for f1, f2, tf1, tf2 in all_skipped:
            print(f"  - {f1} ({tf1} fights) vs {f2} ({tf2} fights)")
        print()

    # ── Evaluation summary (appended) ──
    n = len(all_results)
    if n == 0:
        return

    model_correct_count = sum(1 for r in all_results if r["model_correct"])
    correct_vb = sum(1 for vb in all_valuebets if vb["won"])
    total_vb_stake = sum(vb["stake"] for vb in all_valuebets)
    total_vb_return = sum(vb["stake"] * vb["odds"] for vb in all_valuebets if vb["won"])

    print("=" * 80)
    print("  MODEL EVALUATION")
    print("=" * 80)
    print(f"  Fights evaluated:  {n}")
    print(f"  Model accuracy:    {model_correct_count}/{n}  ({model_correct_count/n*100:.1f}%)")
    print(f"  Value bets:        {len(all_valuebets)}")
    print(f"  VB correct:        {correct_vb}/{len(all_valuebets)}  ({correct_vb/len(all_valuebets)*100:.1f}%)" if all_valuebets else f"  VB correct:        N/A")
    if total_vb_stake > 0:
        profit = total_vb_return - total_vb_stake
        roi = profit / total_vb_stake * 100
        print(f"  VB total stake:    ${total_vb_stake:.2f}")
        print(f"  VB total return:   ${total_vb_return:.2f}")
        print(f"  VB profit / loss:  ${profit:+.2f}")
        print(f"  VB ROI:            {roi:+.2f}%")
        print(f"  Final bankroll:    ${bankroll + profit:.2f}")
    print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if not EVALUATION_FIGHTS:
        print("No fights configured. Edit EVALUATION_FIGHTS in the script.")
    else:
        evaluate_valuebets(EVALUATION_FIGHTS)
