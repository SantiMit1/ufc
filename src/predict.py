import json
import numpy as np
import pandas as pd
import joblib
import shap
from datetime import datetime
from typing import Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from ensemble_utils import ChronologicalStackingEnsemble, ChronologicalStackingEnsembleMultiClass
from stats_utils import shrink_rate, shrink_proportion, _prior_accum_init, _prior_accum_add, _get_current_priors, compute_composite_features


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


def parse_time(time_str: str) -> int:
    if not time_str:
        return 0
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def fight_seconds(round_no: int, time_str: str) -> int:
    if round_no == 0 or not time_str:
        return 0
    return (round_no - 1) * 300 + parse_time(time_str)


def get_stance(fighter_name: str, fighters_cache: dict) -> str:
    entry = fighters_cache.get(fighter_name, {})
    stance = entry.get("stance")
    if stance and stance != "null":
        return stance
    return "Unknown"


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


def update_state(state: dict, fight: dict, is_fighter_1: bool, is_win_loss: bool, win_side: int, finish_type: str | None, opponent_elo: float | None = None) -> None:
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


def compute_stats_from_state(fighter_state: dict, fighter_name: str, fighters_cache: dict, current_date: datetime, category: str = "", priors: dict | None = None) -> dict:
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

    stance = get_stance(fighter_name, fighters_cache)
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
            td_avg_per_15min = shrink_rate(td_landed, total_minutes, pm / 15.0, total_fights=total_fights) * 15.0
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

    feat_dict = {
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
    composites = compute_composite_features(feat_dict)
    feat_dict.update(composites)
    return feat_dict


def build_fighter_states(fights: list, fighters_cache: dict) -> dict:
    for fight in fights:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")

    filtered = sorted([f for f in fights if f["_parsed_date"] >= CUTOFF_DATE], key=lambda f: f["_parsed_date"])
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

        is_win_loss, win_side, finish_type = classify_method(fight["method"], fight["winner"], f1, f2)
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


def find_fighter(query: str, fighter_states: dict, fighters_cache: dict) -> list:
    q = query.lower().strip()
    names = set()
    for name in fighter_states:
        if q in name.lower():
            names.add(name)
    for name in fighters_cache:
        if q in name.lower():
            names.add(name)
    return sorted(names)


def select_fighter(prompt: str, fighter_states: dict, fighters_cache: dict) -> str:
    while True:
        query = input(f"\n{prompt}: ").strip()
        if not query:
            print("  Enter a name.")
            continue

        matches = find_fighter(query, fighter_states, fighters_cache)

        if len(matches) == 0:
            print(f"  No fighters found matching '{query}'.")
            continue

        if len(matches) == 1:
            print(f"  Selected: {matches[0]}")
            return matches[0]

        print(f"\n  Multiple matches for '{query}':")
        for i, name in enumerate(matches, 1):
            has = "fights" if name in fighter_states else "no fights"
            print(f"    {i}. {name} ({has})")
        print(f"    {len(matches) + 1}. Try a different name")

        choice = input("\n  Select number: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(matches):
                return matches[idx]
        except ValueError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--features", default=FEATURE_COLS_PATH)
    parser.add_argument("--method-model", default=None)
    parser.add_argument("--round-model", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  UFC FIGHT PREDICTOR")
    print("=" * 60)

    print("\nLoading data...")
    with open(FIGHTS_PATH) as f:
        fights = json.load(f)
    print(f"  {len(fights)} fights loaded")

    with open(FIGHTERS_CACHE_PATH) as f:
        fighters_cache = json.load(f)
    print(f"  {len(fighters_cache)} fighters in cache")

    print("  Computing population priors by weight class (incremental, no lookahead)...")
    _prior_accum_init()
    # Parse dates first
    for fight in fights:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    # Sort and add to accumulator
    for fight in sorted(fights, key=lambda f: f["_parsed_date"]):
        if fight["_parsed_date"] >= CUTOFF_DATE:
            _prior_accum_add(fight)
    priors = _get_current_priors()
    for cat, vals in priors.items():
        print(f"    {cat}: sig_str/min={vals['sig_str_landed_per_min']:.2f}, td/15min={vals['td_avg_per_15min']:.2f}")

    print("  Loading model...")
    model = joblib.load(args.model)
    feature_meta = joblib.load(args.features)
    model_type = feature_meta.get("model_type", "stacking")

    if model_type == "lightgbm":
        print(f"  Model type: LightGBM")
    elif model_type == "stacking":
        print(f"  Model type: Stacked ensemble")
    else:
        print(f"  Model type: Logistic Regression")
    print("  Model loaded")

    method_model = None
    round_model = None
    try:
        method_model = joblib.load(args.method_model) if hasattr(args, 'method_model') and args.method_model else None
    except Exception:
        pass
    if method_model is None:
        try:
            method_model = joblib.load(METHOD_MODEL_PATH)
        except Exception:
            pass
    try:
        round_model = joblib.load(args.round_model) if hasattr(args, 'round_model') and args.round_model else None
    except Exception:
        pass
    if round_model is None:
        try:
            round_model = joblib.load(ROUND_MODEL_PATH)
        except Exception:
            pass
    if method_model is not None:
        print("  Method model loaded")
    if round_model is not None:
        print("  Round model loaded")

    shap_explainer = None
    if model_type in ("lightgbm", "stacking"):
        lgb_base = None
        if model_type == "stacking":
            lgb_base = model.named_estimators_.get("lgbm")
        else:
            if hasattr(model, 'calibrated_classifiers_'):
                cc = model.calibrated_classifiers_[0]
                lgb_base = cc.estimator if hasattr(cc, 'estimator') else cc.base_estimator_
            else:
                lgb_base = model
        if lgb_base is not None and hasattr(lgb_base, "named_steps"):
            lgb_base = lgb_base.named_steps.get("logreg", lgb_base)
        if lgb_base is not None:
            shap_explainer = shap.TreeExplainer(lgb_base)
            if model_type == "stacking":
                print("  SHAP explainer ready (LightGBM base of stacked ensemble)")
            else:
                print("  SHAP explainer ready")

    print("  Building fighter histories...")
    fighter_states = build_fighter_states(fights, fighters_cache)
    print(f"  {len(fighter_states)} fighters with fight history")

    fighter_a = select_fighter("Enter first fighter name", fighter_states, fighters_cache)
    fighter_b = select_fighter("Enter second fighter name", fighter_states, fighters_cache)

    if fighter_a == fighter_b:
        print("\nBoth fighters are the same. Exiting.")
        return

    # Weight class selection
    weight_classes = [
        "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
        "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
        "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
        "Women's Featherweight", "Catch Weight",
    ]
    print("\n  Weight class:")
    for i, wc in enumerate(weight_classes, 1):
        print(f"    {i}. {wc}")
    wc_input = input("\n  Select (number or name): ").strip()
    try:
        category = weight_classes[int(wc_input) - 1]
    except (ValueError, IndexError):
        category = wc_input
    print(f"  Weight class: {category}")

    # Max rounds selection
    print("\n  Rounds:")
    print("    1. 3 rounds (non-title / prelim)")
    print("    2. 5 rounds (title / main event)")
    r_input = input("  Select (1/2, Enter=3): ").strip()
    max_rounds = 5 if r_input == "2" else 3
    print(f"  Max rounds: {max_rounds}")

    # Predict
    print(f"\n  {'=' * 56}")
    print(f"  {fighter_a}  vs  {fighter_b}")
    print(f"  {'=' * 56}")
    print(f"  Weight class: {category}")

    current_date = datetime.now()

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

        row["age_a"] = feat1["age"]
        row["age_b"] = feat2["age"]
        row["stance_a"] = feat1["stance"]
        row["stance_b"] = feat2["stance"]
        row["category"] = category
        row["age_diff"] = safe_sub(feat1["age"], feat2["age"])
        row["height_diff"] = safe_sub(height1, height2)
        row["reach_diff"] = safe_sub(reach1, reach2)
        row["elo_diff"] = safe_sub(feat1["elo"], feat2["elo"])

        row["striking_strength_diff"] = safe_sub(feat1["striking"], feat2["striking"])
        row["grappling_strength_diff"] = safe_sub(feat1["grappling"], feat2["grappling"])
        row["durability_diff"] = safe_sub(feat1["durability"], feat2["durability"])
        row["momentum_diff"] = safe_sub(feat1["momentum"], feat2["momentum"])
        row["experience_diff"] = safe_sub(feat1["experience"], feat2["experience"])

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

        shap_vals = None
        if shap_explainer is not None:
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore",
                    message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
                )
                sv = shap_explainer.shap_values(X_encoded)
            if isinstance(sv, list):
                shap_vals = sv[1][0]
            else:
                shap_vals = sv[0]
        return prob, shap_vals, method_proba, round_proba

    # Predict in both orders and average to remove order-dependent bias
    prob_a_forward, shap_a_forward, method_a_forward, round_a_forward = predict_order(fighter_a, fighter_b)
    prob_b_forward, shap_b_forward, method_b_forward, round_b_forward = predict_order(fighter_b, fighter_a)
    prob_a = (prob_a_forward + (1.0 - prob_b_forward)) / 2.0
    prob_b = 1.0 - prob_a

    method_proba = None
    round_proba = None
    if method_a_forward is not None and method_b_forward is not None:
        method_proba = (method_a_forward + method_b_forward) / 2.0
    if round_a_forward is not None and round_b_forward is not None:
        round_proba = (round_a_forward + round_b_forward) / 2.0

    height_a, reach_a = get_phys(fighter_a, "height_cm"), get_phys(fighter_a, "reach_cm")
    height_b, reach_b = get_phys(fighter_b, "height_cm"), get_phys(fighter_b, "reach_cm")
    state_a = fighter_states.get(fighter_a, make_initial_state())
    state_b = fighter_states.get(fighter_b, make_initial_state())
    feat_a = compute_stats_from_state(state_a, fighter_a, fighters_cache, current_date, category=category, priors=priors)
    feat_b = compute_stats_from_state(state_b, fighter_b, fighters_cache, current_date, category=category, priors=priors)

    favorite, underdog = (fighter_a, fighter_b) if prob_a >= prob_b else (fighter_b, fighter_a)
    fav_prob, dog_prob = (prob_a, prob_b) if prob_a >= prob_b else (prob_b, prob_a)

    print(f"\n  PREDICTION")
    print(f"  {'-' * 56}")
    print(f"\n  Favorite: {favorite} ({fav_prob * 100:.1f}%, Fair Odds: {1.0 / fav_prob:.2f})")
    print(f"  Underdog: {underdog} ({dog_prob * 100:.1f}%, Fair Odds: {1.0 / dog_prob:.2f})")

    # ─── COMPARATIVE TABLE ──────────────────────────────────────────────────────
    sw, vw = 24, 20
    line_fmt = "  {:<" + str(sw) + "} {:<" + str(vw) + "} {:<" + str(vw) + "} {:<" + str(vw) + "}"
    sep = "  " + "-" * (sw + vw * 3 + 4)
    eq_sep = "  " + "=" * (sw + vw * 3 + 4)
    print(f"\n{eq_sep}")
    print(f"  {'COMPARATIVE TABLE':^{sw + vw * 3 + 4}}")
    print(eq_sep)
    print(line_fmt.format("Stat", fighter_a, fighter_b, "Diff"))
    print(sep)

    def _ff(v, spec=".2f"):
        if isinstance(v, (int, float)) and not np.isnan(v):
            return f"{v:{spec}}"
        return "-"

    def _line(label, va, vb, spec=".2f"):
        if isinstance(va, str):
            da, db, diff = va, vb, ""
        else:
            da, db = _ff(va, spec), _ff(vb, spec)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not np.isnan(va) and not np.isnan(vb):
                diff = _ff(va - vb, "+.2f" if "." in spec else "+.0f")
            else:
                diff = ""
        print(line_fmt.format(label, da, db, diff))

    _line("Height (cm)", height_a, height_b, ".0f")
    _line("Reach (cm)", reach_a, reach_b, ".0f")
    _line("Age (years)", feat_a["age"], feat_b["age"])
    _line("Stance", feat_a["stance"], feat_b["stance"])
    _line("Record", f"{feat_a['wins']}W-{feat_a['losses']}L ({feat_a['total_fights']})",
          f"{feat_b['wins']}W-{feat_b['losses']}L ({feat_b['total_fights']})")
    _line("Elo", feat_a["elo"], feat_b["elo"], ".0f")
    _line("Win%", feat_a["win_pct"] * 100 if not np.isnan(feat_a["win_pct"]) else np.nan,
          feat_b["win_pct"] * 100 if not np.isnan(feat_b["win_pct"]) else np.nan, ".1f")
    _line("Sig. Str. Landed/min", feat_a["sig_str_landed_per_min"], feat_b["sig_str_landed_per_min"])
    _line("Decay Sig. Str./min", feat_a["decay_sig_per_min"], feat_b["decay_sig_per_min"])
    _line("Sig. Str. Absorbed/min", feat_a["sig_str_absorbed_per_min"], feat_b["sig_str_absorbed_per_min"])
    _line("Decay Sig. Str. Absorbed/min", feat_a["decay_sig_absorbed_per_min"], feat_b["decay_sig_absorbed_per_min"])
    _line("Takedowns/15min", feat_a["td_avg_per_15min"], feat_b["td_avg_per_15min"])
    _line("Decay Takedowns/15min", feat_a["decay_td_per_15min"], feat_b["decay_td_per_15min"])
    _line("TD Accuracy (%)", feat_a["td_accuracy"] * 100 if not np.isnan(feat_a["td_accuracy"]) else np.nan,
          feat_b["td_accuracy"] * 100 if not np.isnan(feat_b["td_accuracy"]) else np.nan, ".1f")
    _line("TD Defense (%)", feat_a["td_defense"] * 100 if not np.isnan(feat_a["td_defense"]) else np.nan,
          feat_b["td_defense"] * 100 if not np.isnan(feat_b["td_defense"]) else np.nan, ".1f")
    _line("KO Loss Rate (last 5) (%)",
          feat_a["recent_5_ko_loss_rate"] * 100 if not np.isnan(feat_a["recent_5_ko_loss_rate"]) else np.nan,
          feat_b["recent_5_ko_loss_rate"] * 100 if not np.isnan(feat_b["recent_5_ko_loss_rate"]) else np.nan, ".0f")
    _line("Last 5", f"{feat_a['recent_5_wins']}W-{feat_a['recent_5_losses']}L",
          f"{feat_b['recent_5_wins']}W-{feat_b['recent_5_losses']}L")
    _line("Avg Opp Elo (wins)", feat_a["avg_opp_elo_wins"], feat_b["avg_opp_elo_wins"], ".0f")
    streak_a = (f"{feat_a['current_win_streak']}W streak" if feat_a["current_win_streak"] > 0
                else f"{feat_a['current_losing_streak']}L streak" if feat_a["current_losing_streak"] > 0
                else "-")
    streak_b = (f"{feat_b['current_win_streak']}W streak" if feat_b["current_win_streak"] > 0
                else f"{feat_b['current_losing_streak']}L streak" if feat_b["current_losing_streak"] > 0
                else "-")
    _line("Streak", streak_a, streak_b)
    print(sep)
    _line("Striking Strength", feat_a["striking"], feat_b["striking"])
    _line("Grappling Strength", feat_a["grappling"], feat_b["grappling"])
    _line("Durability", feat_a["durability"], feat_b["durability"])
    _line("Momentum", feat_a["momentum"], feat_b["momentum"])
    _line("Experience (opp quality)", feat_a["experience"], feat_b["experience"])
    print(eq_sep)
    print()

    # ─── FINISH METHOD PREDICTION ───────────────────────────────────────────────
    if method_proba is not None:
        method_labels = ["KO", "SUB", "DEC"]
        print(f"\n  FINISH METHOD PREDICTION")
        print(f"  {'-' * 56}")
        for label, p in zip(method_labels, method_proba):
            print(f"    {label:<6s}  {p * 100:5.1f}%")
        # Most likely method
        best_method_idx = method_proba.argmax()
        print(f"  -> Predicted finish: {method_labels[best_method_idx]} ({method_proba[best_method_idx] * 100:.1f}%)")
        print()

    # ─── ROUND PREDICTION ───────────────────────────────────────────────────────
    if round_proba is not None:
        is_decision = method_proba is not None and method_proba.argmax() == 2
        if is_decision:
            print(f"  ROUND PREDICTION (max rounds: {max_rounds})")
            print(f"  {'-' * 56}")
            print(f"    -> Goes to decision - always round {max_rounds}")
            print()
        else:
            round_proba = constrain_round_probas(round_proba, max_rounds)
            print(f"  ROUND PREDICTION (max rounds: {max_rounds})")
            print(f"  {'-' * 56}")
            for r, p in enumerate(round_proba, 1):
                print(f"    Round {r:<2d}  {p * 100:5.1f}%")
            expected_round = sum((r + 1) * p for r, p in enumerate(round_proba))
            best_round_idx = round_proba.argmax()
            print(f"  -> Most likely: Round {best_round_idx + 1} ({round_proba[best_round_idx] * 100:.1f}%)")
            print(f"  -> Expected round: {expected_round:.2f}")
            print()

    # ─── SHAP EXPLANATION ───────────────────────────────────────────────────────
    if shap_explainer is not None and shap_a_forward is not None:
        feat_names = feature_meta["feature_cols_final"]
        shap_fav = shap_a_forward if prob_a >= prob_b else shap_b_forward

        pairs = list(zip(feat_names, shap_fav))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"  SHAP - {favorite} (fav) vs {underdog} (dog)")
        print(f"  {'-' * 56}")
        for feat, val in pairs[:8]:
            favor = favorite if val > 0 else underdog
            print(f"    {feat:<42s} {val:+7.4f}  -> {favor}")
        print()


def constrain_round_probas(probas, max_rounds):
    if probas is None or max_rounds >= 5:
        return probas
    probas = probas.copy()
    probas[max_rounds:] = 0.0
    total = probas.sum()
    return probas / total if total > 0 else probas


def safe_sub(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (np.isnan(a) or np.isnan(b)):
            return a - b
    return np.nan


if __name__ == "__main__":
    main()
