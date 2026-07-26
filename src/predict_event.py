"""
Predict all fights for a given UFC event.

Usage:
  .venv/Scripts/python src/predict_event.py --event "UFC 328: Chimaev vs. Strickland"
  .venv/Scripts/python src/predict_event.py --event "UFC Fight Night: Fiziev vs. Torres"

Outputs JSON with each fight's prediction probabilities.
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any

from ensemble_utils import ChronologicalStackingEnsemble, ChronologicalStackingEnsembleMultiClass
from stats_utils import shrink_rate, shrink_proportion, _prior_accum_init, _prior_accum_add, _get_current_priors


# ── Paths (same as predict.py) ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FIGHTS_PATH = BASE_DIR / "data" / "fights.json"
FIGHTERS_CACHE_PATH = BASE_DIR / "data" / "fighters_cache.json"
MODEL_PATH = BASE_DIR / "models" / "ufc_stacking_ensemble.pkl"
FEATURE_COLS_PATH = BASE_DIR / "models" / "ufc_stacking_ensemble_meta.pkl"
METHOD_MODEL_PATH = BASE_DIR / "models" / "ufc_method_model.pkl"
ROUND_MODEL_PATH = BASE_DIR / "models" / "ufc_round_model.pkl"

CUTOFF_DATE = datetime(2001, 1, 1)
ELO_K = 96
ELO_INITIAL = 1500


# ── Helper functions (copied from predict.py) ────────────────────────────────
def parse_time(time_str: str) -> int:
    if not time_str:
        return 0
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def fight_seconds(round_no: int, time_str: str) -> int:
    if round_no == 0 or not time_str:
        return 0
    return (round_no - 1) * 300 + parse_time(time_str)


def classify_method(method: str, winner: str, fighter_1: str, fighter_2: str) -> tuple:
    if winner in ("Draw", "No Contest") or "DQ" in method:
        return False, 0, None
    if winner == fighter_1:
        win_side = 1
    elif winner == fighter_2:
        win_side = 2
    else:
        return False, 0, None
    if method.startswith("KO/TKO"):
        ft = "KO"
    elif method.startswith("SUB"):
        ft = "SUB"
    else:
        ft = "DEC"
    return True, win_side, ft


def get_k_factor(total_fights: int) -> float:
    if total_fights <= 5:
        return 96.0
    elif total_fights <= 10:
        return 64.0
    elif total_fights <= 20:
        return 40.0
    else:
        return 24.0


def apply_elo_decay(elo: float, last_fight_date: datetime | None, fight_date: datetime) -> float:
    if last_fight_date is None:
        return elo
    days_inactive = (fight_date - last_fight_date).days
    if days_inactive <= 365:
        return elo
    years_inactive = days_inactive / 365.0
    decay_ratio = min(0.85, (years_inactive - 1.0) * 0.25)
    return elo - (elo - ELO_INITIAL) * decay_ratio


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a: float, rating_b: float, score_a: float, k_a: float = ELO_K, k_b: float = ELO_K) -> tuple:
    exp_a = elo_expected(rating_a, rating_b)
    return rating_a + k_a * (score_a - exp_a), rating_b + k_b * ((1.0 - score_a) - (1.0 - exp_a))


def make_initial_state() -> dict:
    return {
        "total_fights": 0, "wins": 0, "losses": 0, "draws": 0, "no_contests": 0,
        "current_win_streak": 0, "current_losing_streak": 0,
        "wins_by_ko": 0, "wins_by_sub": 0, "wins_by_dec": 0,
        "losses_by_ko": 0, "losses_by_sub": 0, "losses_by_dec": 0,
        "sig_str_landed": 0, "sig_str_attempted": 0,
        "total_str_landed": 0, "total_str_attempted": 0,
        "td_landed": 0, "td_attempted": 0, "sub_attempts": 0,
        "control_time_seconds": 0, "knockdowns": 0,
        "sig_str_absorbed": 0,
        "td_against_landed": 0, "td_against_attempted": 0,
        "total_seconds_fought": 0, "last_fight_date": None,
        "elo": ELO_INITIAL,
        "recent_fights": [],
        "sum_opp_elo": 0.0,
        "count_opp_faced": 0,
        "sum_opp_elo_wins": 0.0,
        "count_opp_wins": 0,
    }


def safe_int(val: Any) -> int:
    return int(val) if val is not None else 0


def safe_sub(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (np.isnan(a) or np.isnan(b)):
            return a - b
    return np.nan


def update_state(state: dict, fight: dict, is_fighter_1: bool, is_win_loss: bool,
                 win_side: int, finish_type: str | None,
                 opponent_elo: float | None = None) -> None:
    stats = fight.get("stats_fighter_1", {}) if is_fighter_1 else fight.get("stats_fighter_2", {})
    opp_stats = fight.get("stats_fighter_2", {}) if is_fighter_1 else fight.get("stats_fighter_1", {})

    state["total_fights"] += 1

    won = None
    if is_win_loss:
        fighter_won = (is_fighter_1 and win_side == 1) or (not is_fighter_1 and win_side == 2)
        won = fighter_won
        if fighter_won:
            state["wins"] += 1
            state["current_win_streak"] += 1
            state["current_losing_streak"] = 0
            if finish_type == "KO":
                state["wins_by_ko"] += 1
            elif finish_type == "SUB":
                state["wins_by_sub"] += 1
            elif finish_type == "DEC":
                state["wins_by_dec"] += 1
        else:
            state["losses"] += 1
            state["current_losing_streak"] += 1
            state["current_win_streak"] = 0
            if finish_type == "KO":
                state["losses_by_ko"] += 1
            elif finish_type == "SUB":
                state["losses_by_sub"] += 1
            elif finish_type == "DEC":
                state["losses_by_dec"] += 1
    else:
        if fight["winner"] == "Draw":
            state["draws"] += 1
        else:
            state["no_contests"] += 1

    sig_landed = safe_int(stats.get("sig_strikes", {}).get("landed"))
    sig_attempted = safe_int(stats.get("sig_strikes", {}).get("attempted"))
    total_landed = safe_int(stats.get("total_strikes", {}).get("landed"))
    total_attempted = safe_int(stats.get("total_strikes", {}).get("attempted"))

    state["sig_str_landed"] += sig_landed
    state["sig_str_attempted"] += sig_attempted
    state["total_str_landed"] += total_landed
    state["total_str_attempted"] += total_attempted

    td_landed = safe_int(stats.get("takedowns", {}).get("landed"))
    td_attempted = safe_int(stats.get("takedowns", {}).get("attempted"))
    opp_td_landed = safe_int(opp_stats.get("takedowns", {}).get("landed"))
    opp_td_attempted = safe_int(opp_stats.get("takedowns", {}).get("attempted"))

    state["td_landed"] += td_landed
    state["td_attempted"] += td_attempted
    state["sub_attempts"] += safe_int(stats.get("sub_attempts"))
    state["control_time_seconds"] += safe_int(stats.get("control_time_seconds"))
    state["knockdowns"] += safe_int(stats.get("knockdowns"))
    state["sig_str_absorbed"] += safe_int(opp_stats.get("sig_strikes", {}).get("landed"))
    state["td_against_landed"] += opp_td_landed
    state["td_against_attempted"] += opp_td_attempted
    fight_secs = fight_seconds(fight.get("round", 0), fight.get("time", ""))
    state["total_seconds_fought"] += fight_secs
    state["last_fight_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")

    # --- Recent fights tracking ---
    ko_loss = won is False and finish_type == "KO"
    sub_loss = won is False and finish_type == "SUB"
    fight_date = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    opp_sig_landed = safe_int(opp_stats.get("sig_strikes", {}).get("landed"))
    fight_record = {
        "date": fight_date,
        "sig_landed": sig_landed,
        "sig_attempted": sig_attempted,
        "sig_absorbed": opp_sig_landed,
        "td_landed": td_landed,
        "td_attempted": td_attempted,
        "sub_attempts": safe_int(stats.get("sub_attempts")),
        "total_seconds": fight_secs,
        "won": won,
        "ko_loss": ko_loss,
        "sub_loss": sub_loss,
    }
    state["recent_fights"].append(fight_record)
    if len(state["recent_fights"]) > 5:
        state["recent_fights"].pop(0)

    # --- Opponent quality ---
    if opponent_elo is not None:
        state["sum_opp_elo"] += opponent_elo
        state["count_opp_faced"] += 1
        if won is True:
            state["sum_opp_elo_wins"] += opponent_elo
            state["count_opp_wins"] += 1


def compute_stats_from_state(fighter_state: dict, fighter_name: str,
                             fighters_cache: dict, current_date: datetime,
                             category: str = "", priors: dict | None = None) -> dict:
    f = fighter_state
    total_fights = f["total_fights"]
    wins = f["wins"]
    losses = f["losses"]

    entry = fighters_cache.get(fighter_name, {})
    dob_str = entry.get("dob")
    if dob_str:
        age = (current_date - datetime.strptime(dob_str, "%Y-%m-%d")).days / 365.25
    else:
        age = np.nan

    stance = entry.get("stance", "Unknown") if entry.get("stance") and entry["stance"] != "null" else "Unknown"
    win_pct = wins / total_fights if total_fights > 0 else np.nan
    ko_rate = f["wins_by_ko"] / wins if wins > 0 else np.nan
    sub_rate = f["wins_by_sub"] / wins if wins > 0 else np.nan
    dec_rate = f["wins_by_dec"] / wins if wins > 0 else np.nan
    ko_loss_rate = f["losses_by_ko"] / losses if losses > 0 else np.nan
    sub_loss_rate = f["losses_by_sub"] / losses if losses > 0 else np.nan

    total_seconds = f["total_seconds_fought"]
    total_minutes = total_seconds / 60.0 if total_seconds > 0 else np.nan

    if priors is not None:
        p = priors.get(category, priors.get("global", {}))
    else:
        p = {}

    sig_str_landed = f["sig_str_landed"]
    sig_str_attempted = f["sig_str_attempted"]
    sig_str_absorbed = f["sig_str_absorbed"]

    if total_minutes > 0 and not np.isnan(total_minutes):
        pm = p.get("sig_str_landed_per_min", np.nan)
        sig_str_landed_per_min = shrink_rate(sig_str_landed, total_minutes, pm, total_fights=total_fights) if not np.isnan(pm) else sig_str_landed / total_minutes
        pm = p.get("sig_str_absorbed_per_min", np.nan)
        sig_str_absorbed_per_min = shrink_rate(sig_str_absorbed, total_minutes, pm, total_fights=total_fights) if not np.isnan(pm) else sig_str_absorbed / total_minutes
    else:
        sig_str_landed_per_min = np.nan
        sig_str_absorbed_per_min = np.nan

    pa = p.get("sig_str_accuracy", np.nan)
    sig_str_accuracy = shrink_proportion(sig_str_landed, sig_str_attempted, pa, total_fights=total_fights) if not np.isnan(pa) else (sig_str_landed / sig_str_attempted if sig_str_attempted > 0 else np.nan)

    td_landed = f["td_landed"]
    td_attempted = f["td_attempted"]
    td_against_landed = f["td_against_landed"]
    td_against_attempted = f["td_against_attempted"]

    if total_minutes > 0 and not np.isnan(total_minutes):
        pm = p.get("td_avg_per_15min", np.nan)
        if not np.isnan(pm):
            td_avg_per_15min = shrink_rate(td_landed, total_minutes, pm, total_fights=total_fights) * 15.0
        else:
            td_avg_per_15min = td_landed / total_minutes * 15.0
    else:
        td_avg_per_15min = np.nan

    pa = p.get("td_accuracy", np.nan)
    td_accuracy = shrink_proportion(td_landed, td_attempted, pa, total_fights=total_fights) if not np.isnan(pa) else (td_landed / td_attempted if td_attempted > 0 else np.nan)
    pd_ = p.get("td_defense", np.nan)
    if not np.isnan(pd_):
        td_def = td_against_attempted - td_against_landed
        td_defense = shrink_proportion(td_def, td_against_attempted, pd_, total_fights=total_fights)
    else:
        td_defense = (1.0 - td_against_landed / td_against_attempted) if td_against_attempted > 0 else np.nan
    sub_att = f["sub_attempts"]
    sub_att_per_15min = sub_att / total_minutes * 15.0 if total_minutes > 0 and not np.isnan(total_minutes) else np.nan
    ctrl_time_pct = f["control_time_seconds"] / total_seconds if total_seconds > 0 else np.nan

    days_since_last_fight = np.nan
    if f["last_fight_date"] is not None:
        days_since_last_fight = (current_date - f["last_fight_date"]).days

    # --- Recent form ---
    recent = f["recent_fights"]
    last_3 = recent[-3:] if len(recent) >= 3 else recent
    last_5 = recent[-5:] if len(recent) >= 5 else recent

    recent_3_wins = sum(1 for r in last_3 if r["won"] is True)
    recent_3_losses = sum(1 for r in last_3 if r["won"] is False)
    recent_5_wins = sum(1 for r in last_5 if r["won"] is True)
    recent_5_losses = sum(1 for r in last_5 if r["won"] is False)

    recent_3_ko_losses = sum(1 for r in last_3 if r["ko_loss"])
    recent_5_ko_losses = sum(1 for r in last_5 if r["ko_loss"])

    recent_3_ko_loss_rate = recent_3_ko_losses / len(last_3) if len(last_3) > 0 else np.nan
    recent_5_ko_loss_rate = recent_5_ko_losses / len(last_5) if len(last_5) > 0 else np.nan

    # --- Decay-weighted stats (lambda = 0.5 per year) ---
    LAMBDA = 0.5
    total_w = 0.0
    w_sig = 0.0
    w_sig_abs = 0.0
    w_td = 0.0
    w_sec = 0.0
    for r in recent:
        years_ago = (current_date - r["date"]).days / 365.25
        w = np.exp(-LAMBDA * years_ago)
        total_w += w
        w_sig += r["sig_landed"] * w
        w_sig_abs += r["sig_absorbed"] * w
        w_td += r["td_landed"] * w
        w_sec += r["total_seconds"] * w

    decay_sig_per_min = w_sig / (w_sec / 60.0) if w_sec > 0 else np.nan
    decay_sig_absorbed_per_min = w_sig_abs / (w_sec / 60.0) if w_sec > 0 else np.nan
    decay_td_per_15min = w_td / (w_sec / 60.0) * 15.0 if w_sec > 0 else np.nan

    # --- Opponent quality ---
    count_faced = f["count_opp_faced"]
    count_wins = f["count_opp_wins"]
    avg_opp_elo = f["sum_opp_elo"] / count_faced if count_faced > 0 else np.nan
    avg_opp_elo_wins = f["sum_opp_elo_wins"] / count_wins if count_wins > 0 else np.nan

    return {
        "age": age,
        "stance": stance,
        "win_pct": win_pct,
        "ko_rate": ko_rate,
        "sub_rate": sub_rate,
        "dec_rate": dec_rate,
        "ko_loss_rate": ko_loss_rate,
        "sub_loss_rate": sub_loss_rate,
        "sig_str_landed_per_min": sig_str_landed_per_min,
        "sig_str_absorbed_per_min": sig_str_absorbed_per_min,
        "sig_str_accuracy": sig_str_accuracy,
        "td_avg_per_15min": td_avg_per_15min,
        "td_accuracy": td_accuracy,
        "td_defense": td_defense,
        "sub_att_per_15min": sub_att_per_15min,
        "ctrl_time_pct": ctrl_time_pct,
        "days_since_last_fight": days_since_last_fight,
        "current_win_streak": f["current_win_streak"],
        "current_losing_streak": f["current_losing_streak"],
        "total_fights": total_fights,
        "is_debut": total_fights == 0,
        "elo": f["elo"],
        "wins": wins,
        "losses": losses,
        "recent_3_wins": recent_3_wins,
        "recent_3_losses": recent_3_losses,
        "recent_5_wins": recent_5_wins,
        "recent_5_losses": recent_5_losses,
        "recent_3_ko_loss_rate": recent_3_ko_loss_rate,
        "recent_5_ko_loss_rate": recent_5_ko_loss_rate,
        "decay_sig_per_min": decay_sig_per_min,
        "decay_sig_absorbed_per_min": decay_sig_absorbed_per_min,
        "decay_td_per_15min": decay_td_per_15min,
        "avg_opp_elo": avg_opp_elo,
        "avg_opp_elo_wins": avg_opp_elo_wins,
    }


def build_fighter_states(fights: list, fighters_cache: dict) -> dict:
    for fight in fights:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")

    filtered = sorted(
        [f for f in fights if f["_parsed_date"] >= CUTOFF_DATE],
        key=lambda f: f["_parsed_date"],
    )
    fighter_state: dict[str, dict] = {}

    for fight in filtered:
        f1, f2 = fight["fighter_1"], fight["fighter_2"]
        if f1 not in fighter_state:
            fighter_state[f1] = make_initial_state()
        if f2 not in fighter_state:
            fighter_state[f2] = make_initial_state()

        # Apply Elo decay for inactivity (>1 year)
        fighter_state[f1]["elo"] = apply_elo_decay(fighter_state[f1]["elo"], fighter_state[f1]["last_fight_date"], fight["_parsed_date"])
        fighter_state[f2]["elo"] = apply_elo_decay(fighter_state[f2]["elo"], fighter_state[f2]["last_fight_date"], fight["_parsed_date"])

        is_win_loss, win_side, finish_type = classify_method(
            fight["method"], fight["winner"], f1, f2
        )
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
            f1_new, f2_new = elo_update(fighter_state[f1]["elo"],
                                        fighter_state[f2]["elo"], score_a,
                                        k_a=k_a, k_b=k_b)
            fighter_state[f1]["elo"] = f1_new
            fighter_state[f2]["elo"] = f2_new

    return fighter_state


def predict_fight(fighter_a: str, fighter_b: str, category: str,
                  fighter_states: dict, fighters_cache: dict,
                  model, feature_meta: dict, current_date: datetime,
                  priors: dict | None = None,
                  method_model=None, round_model=None) -> dict:
    """Predict a single fight and return probabilities."""
    def get_phys(name, key):
        v = fighters_cache.get(name, {}).get(key)
        return float(v) if v is not None else np.nan

    def predict_order(f1, f2):
        height1, reach1 = get_phys(f1, "height_cm"), get_phys(f1, "reach_cm")
        height2, reach2 = get_phys(f2, "height_cm"), get_phys(f2, "reach_cm")

        state1 = fighter_states.get(f1, make_initial_state())
        state2 = fighter_states.get(f2, make_initial_state())

        feat1 = compute_stats_from_state(state1, f1, fighters_cache, current_date, category=category, priors=priors)
        feat2 = compute_stats_from_state(state2, f2, fighters_cache, current_date, category=category, priors=priors)

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

    # Predict in both orders and average to remove order-dependent bias
    prob_a_forward, method_a_forward, round_a_forward = predict_order(fighter_a, fighter_b)
    prob_b_forward, method_b_forward, round_b_forward = predict_order(fighter_b, fighter_a)
    prob_a = (prob_a_forward + (1.0 - prob_b_forward)) / 2.0
    prob_b = 1.0 - prob_a

    result = {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "prob_a": round(float(prob_a), 6),
        "prob_b": round(float(prob_b), 6),
        "category": category,
    }

    if method_a_forward is not None and method_b_forward is not None:
        method_proba = (method_a_forward + method_b_forward) / 2.0
        result["method_probabilities"] = {
            "KO": round(float(method_proba[0]), 6),
            "SUB": round(float(method_proba[1]), 6),
            "DEC": round(float(method_proba[2]), 6),
        }
        result["predicted_method"] = ["KO", "SUB", "DEC"][int(method_proba.argmax())]

    if round_a_forward is not None and round_b_forward is not None:
        round_proba = (round_a_forward + round_b_forward) / 2.0
        result["round_probabilities"] = {
            str(i + 1): round(float(round_proba[i]), 6) for i in range(5)
        }
        result["predicted_round"] = int(round_proba.argmax()) + 1
        result["expected_round"] = round(float(sum((i + 1) * round_proba[i] for i in range(5))), 4)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict all fights for a UFC event"
    )
    parser.add_argument("--event", required=True,
                        help="Event name, e.g. 'UFC 328: Chimaev vs. Strickland'")
    parser.add_argument("--exact", action="store_true",
                        help="If set, event name must match exactly (otherwise fuzzy)")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--features-path", default=str(FEATURE_COLS_PATH))
    parser.add_argument("--method-model-path", default=None)
    parser.add_argument("--round-model-path", default=None)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    with open(FIGHTS_PATH, encoding="utf-8") as f:
        fights = json.load(f)

    with open(FIGHTERS_CACHE_PATH, encoding="utf-8") as f:
        fighters_cache = json.load(f)

    # ── Find event fights ────────────────────────────────────────────────────
    event_name = args.event.strip()
    if args.exact:
        event_fights = [f for f in fights if f["event_name"] == event_name]
    else:
        # Case-insensitive substring match
        q = event_name.lower()
        event_fights = [f for f in fights if q in f["event_name"].lower()]

    if not event_fights:
        # List matching events for help
        q = event_name.lower()
        all_events = sorted(set(f["event_name"] for f in fights))
        matches = [e for e in all_events if q in e.lower()]
        result = {"error": f"No fights found for event: {event_name}", "matched_events": matches}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

    actual_event_name = event_fights[0]["event_name"]
    event_date = event_fights[0]["event_date"]

    # ── Load model ───────────────────────────────────────────────────────────
    model = joblib.load(args.model_path)
    feature_meta = joblib.load(args.features_path)

    method_model = None
    round_model = None
    try:
        method_model = joblib.load(args.method_model_path) if hasattr(args, 'method_model_path') else None
    except Exception:
        pass
    if method_model is None and METHOD_MODEL_PATH.exists():
        try:
            method_model = joblib.load(METHOD_MODEL_PATH)
        except Exception:
            pass
    try:
        round_model = joblib.load(args.round_model_path) if hasattr(args, 'round_model_path') else None
    except Exception:
        pass
    if round_model is None and ROUND_MODEL_PATH.exists():
        try:
            round_model = joblib.load(ROUND_MODEL_PATH)
        except Exception:
            pass

    # ── Build fighter states (ONLY with fights up to event date) ─────────────
    # ⚠️ CRITICAL: Filter out fights AFTER the event to prevent look-ahead bias.
    # The model must only use statistics known BEFORE the event, just like in
    # the football Poisson model — this is an "in-time simulation".
    event_dt = datetime.strptime(event_date, "%Y-%m-%d")
    historical_fights = [
        f for f in fights
        if datetime.strptime(f["event_date"], "%Y-%m-%d") <= event_dt
    ]
    
    # Compute priors ONLY from historical fights (no lookahead)
    _prior_accum_init()
    for f in sorted(historical_fights, key=lambda x: x.get("event_date", "1970-01-01")):
        _prior_accum_add(f)
    priors = _get_current_priors()

    fighter_states = build_fighter_states(historical_fights, fighters_cache)

    # ── Predict each fight ───────────────────────────────────────────────────
    current_date = event_dt

    predictions = []
    for fight in event_fights:
        f1 = fight["fighter_1"]
        f2 = fight["fighter_2"]
        category = fight["category"]
        winner = fight["winner"]

        try:
            pred = predict_fight(f1, f2, category, fighter_states,
                                 fighters_cache, model, feature_meta, current_date,
                                 priors=priors, method_model=method_model, round_model=round_model)
            pred["winner"] = winner
            predictions.append(pred)
        except Exception as e:
            predictions.append({
                "fighter_a": f1,
                "fighter_b": f2,
                "error": str(e),
                "winner": winner,
            })

    # ── Output ───────────────────────────────────────────────────────────────
    output = {
        "event": actual_event_name,
        "date": event_date,
        "total_fights": len(predictions),
        "predictions": predictions,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
