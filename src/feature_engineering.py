import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any

from stats_utils import compute_priors, shrink_rate, shrink_proportion

FIGHTS_PATH = "data/fights.json"
FIGHTERS_CACHE_PATH = "data/fighters_cache.json"
OUTPUT_PATH = "data/dataset.csv"

CUTOFF_DATE = datetime(2001, 1, 1)
SEED = 42
ELO_K = 96
ELO_INITIAL = 1500


def parse_time(time_str: str) -> int:
    if not time_str:
        return 0
    parts = time_str.split(":")
    minutes = int(parts[0])
    seconds = int(parts[1])
    return minutes * 60 + seconds


def fight_seconds(round_no: int, time_str: str) -> int:
    if round_no == 0 or not time_str:
        return 0
    completed_rounds = round_no - 1
    total_seconds = completed_rounds * 300  # 5 min per round
    total_seconds += parse_time(time_str)
    return total_seconds


def get_stance(fighter_name: str, fighters_cache: dict) -> str:
    entry = fighters_cache.get(fighter_name, {})
    stance = entry.get("stance")
    if stance and stance != "null":
        return stance
    return "Unknown"


def classify_method(method: str, winner: str, fighter_1: str, fighter_2: str) -> tuple:
    """
    Returns (is_win_loss, win_side, finish_type)
    is_win_loss: True if this fight counts as a win/loss for either fighter
    win_side: 1 if fighter_1 wins, 2 if fighter_2 wins, 0 if no winner
    finish_type: 'KO', 'SUB', 'DEC', or None if no win
    """
    if winner == "Draw" or winner == "No Contest":
        return False, 0, None

    if "DQ" in method:
        # DQ counted as No Contest per user spec
        return False, 0, None

    if winner == fighter_1:
        win_side = 1
    elif winner == fighter_2:
        win_side = 2
    else:
        return False, 0, None

    is_win_loss = True
    finish_type = None
    if method.startswith("KO/TKO"):
        finish_type = "KO"
    elif method.startswith("SUB"):
        finish_type = "SUB"
    elif method.endswith("DEC"):
        finish_type = "DEC"
    else:
        finish_type = "DEC"  # fallback for unusual decisions

    return is_win_loss, win_side, finish_type


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


def elo_update(rating_a: float, rating_b: float, score_a: float, k_a: float = ELO_K, k_b: float = ELO_K) -> tuple[float, float]:
    expected_a = elo_expected(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    new_a = rating_a + k_a * (score_a - expected_a)
    new_b = rating_b + k_b * ((1.0 - score_a) - expected_b)
    return new_a, new_b


def compute_stats(
    fighter_state: dict,
    fight: dict,
    is_fighter_1: bool,
    fighters_cache: dict,
    category: str = "",
    priors: dict | None = None,
) -> dict:
    """
    Compute per-fighter features from accumulated state (pre-fight).
    """
    f = fighter_state
    total_fights = f["total_fights"]

    wins = f["wins"]
    losses = f["losses"]

    age_key = "fighter_1_age" if is_fighter_1 else "fighter_2_age"
    age = fight.get(age_key)
    if age is None:
        age = np.nan

    name = fight["fighter_1"] if is_fighter_1 else fight["fighter_2"]
    stance = get_stance(name, fighters_cache)

    win_pct = wins / total_fights if total_fights > 0 else np.nan
    ko_rate = f["wins_by_ko"] / wins if wins > 0 else np.nan
    sub_rate = f["wins_by_sub"] / wins if wins > 0 else np.nan
    dec_rate = f["wins_by_dec"] / wins if wins > 0 else np.nan
    ko_loss_rate = f["losses_by_ko"] / losses if losses > 0 else np.nan
    sub_loss_rate = f["losses_by_sub"] / losses if losses > 0 else np.nan

    total_seconds = f["total_seconds_fought"]
    total_minutes = total_seconds / 60.0 if total_seconds > 0 else np.nan

    sig_str_landed = f["sig_str_landed"]
    sig_str_attempted = f["sig_str_attempted"]
    sig_str_absorbed = f["sig_str_absorbed"]

    if priors is not None:
        p = priors.get(category, priors.get("global", {}))
    else:
        p = {}

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
    if total_minutes > 0 and not np.isnan(total_minutes):
        sub_att_per_15min = sub_att / total_minutes * 15.0
    else:
        sub_att_per_15min = np.nan

    ctrl_time = f["control_time_seconds"]
    if total_seconds > 0:
        ctrl_time_pct = ctrl_time / total_seconds
    else:
        ctrl_time_pct = np.nan

    days_since_last_fight = np.nan
    if f["last_fight_date"] is not None:
        days_since_last_fight = (datetime.strptime(fight["event_date"], "%Y-%m-%d") - f["last_fight_date"]).days

    is_debut = total_fights == 0

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
    fight_date = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    for r in recent:
        years_ago = (fight_date - r["date"]).days / 365.25
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
        "is_debut": is_debut,
        "elo": f["elo"],
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


def update_state(state: dict, fight: dict, is_fighter_1: bool, is_win_loss: bool, win_side: int, finish_type: str | None, opponent_elo: float | None = None) -> None:
    """
    Update accumulated state for a fighter after a fight.
    """
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
        # Draw or No Contest
        if fight["winner"] == "Draw":
            state["draws"] += 1
        else:
            state["no_contests"] += 1

    # Update cumulative stats (always, even for NC/DQ)
    def safe_int(val: Any) -> int:
        if val is None:
            return 0
        return int(val)

    sig_landed = safe_int(stats.get("sig_strikes", {}).get("landed"))
    sig_attempted = safe_int(stats.get("sig_strikes", {}).get("attempted"))
    total_landed = safe_int(stats.get("total_strikes", {}).get("landed"))
    total_attempted = safe_int(stats.get("total_strikes", {}).get("attempted"))
    td_landed = safe_int(stats.get("takedowns", {}).get("landed"))
    td_attempted = safe_int(stats.get("takedowns", {}).get("attempted"))
    sub_attempts = safe_int(stats.get("sub_attempts"))
    ctrl_time = safe_int(stats.get("control_time_seconds"))
    knockdowns = safe_int(stats.get("knockdowns"))

    opp_sig_landed = safe_int(opp_stats.get("sig_strikes", {}).get("landed"))
    opp_td_landed = safe_int(opp_stats.get("takedowns", {}).get("landed"))
    opp_td_attempted = safe_int(opp_stats.get("takedowns", {}).get("attempted"))

    state["sig_str_landed"] += sig_landed
    state["sig_str_attempted"] += sig_attempted
    state["total_str_landed"] += total_landed
    state["total_str_attempted"] += total_attempted
    state["td_landed"] += td_landed
    state["td_attempted"] += td_attempted
    state["sub_attempts"] += sub_attempts
    state["control_time_seconds"] += ctrl_time
    state["knockdowns"] += knockdowns
    state["sig_str_absorbed"] += opp_sig_landed
    state["td_against_landed"] += opp_td_landed
    state["td_against_attempted"] += opp_td_attempted

    fight_secs = fight_seconds(fight.get("round", 0), fight.get("time", ""))
    state["total_seconds_fought"] += fight_secs

    state["last_fight_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")

    # --- Recent fights tracking ---
    ko_loss = won is False and finish_type == "KO"
    sub_loss = won is False and finish_type == "SUB"
    fight_date = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    fight_record = {
        "date": fight_date,
        "sig_landed": sig_landed,
        "sig_attempted": sig_attempted,
        "sig_absorbed": opp_sig_landed,
        "td_landed": td_landed,
        "td_attempted": td_attempted,
        "sub_attempts": sub_attempts,
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


def main():
    random.seed(SEED)

    print("Loading fights...")
    with open(FIGHTS_PATH, "r") as f:
        fights = json.load(f)

    print("Loading fighters cache...")
    with open(FIGHTERS_CACHE_PATH, "r") as f:
        fighters_cache = json.load(f)

    total_raw = len(fights)

    print("Computing population priors by weight class...")
    priors = compute_priors(fights)
    for cat, vals in priors.items():
        print(f"  {cat}: sig_str/min={vals['sig_str_landed_per_min']:.2f}, td/15min={vals['td_avg_per_15min']:.2f}")

    # Parse dates
    for fight in fights:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")

    # Filter by cutoff date
    filtered_fights = [f for f in fights if f["_parsed_date"] >= CUTOFF_DATE]
    discarded = total_raw - len(filtered_fights)
    print(f"Discarded {discarded} fights before {CUTOFF_DATE.date()}")

    # Sort chronologically
    filtered_fights.sort(key=lambda f: f["_parsed_date"])

    # Initialize state dict
    fighter_state: dict[str, dict] = {}

    def make_initial_state() -> dict:
        return {
            "total_fights": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "no_contests": 0,
            "current_win_streak": 0,
            "current_losing_streak": 0,
            "wins_by_ko": 0,
            "wins_by_sub": 0,
            "wins_by_dec": 0,
            "losses_by_ko": 0,
            "losses_by_sub": 0,
            "losses_by_dec": 0,
            "sig_str_landed": 0,
            "sig_str_attempted": 0,
            "total_str_landed": 0,
            "total_str_attempted": 0,
            "td_landed": 0,
            "td_attempted": 0,
            "sub_attempts": 0,
            "control_time_seconds": 0,
            "knockdowns": 0,
            "sig_str_absorbed": 0,
            "td_against_landed": 0,
            "td_against_attempted": 0,
            "total_seconds_fought": 0,
            "last_fight_date": None,
            "elo": ELO_INITIAL,
            "recent_fights": [],
            "sum_opp_elo": 0.0,
            "count_opp_faced": 0,
            "sum_opp_elo_wins": 0.0,
            "count_opp_wins": 0,
        }

    rows = []
    debut_a_count = 0
    debut_b_count = 0
    unknown_stance_count = 0

    for idx, fight in enumerate(filtered_fights, 1):
        f1_name = fight["fighter_1"]
        f2_name = fight["fighter_2"]

        # Ensure both fighters have state entries
        if f1_name not in fighter_state:
            fighter_state[f1_name] = make_initial_state()
        if f2_name not in fighter_state:
            fighter_state[f2_name] = make_initial_state()

        # Apply Elo decay for inactivity (>1 year)
        f1_state = fighter_state[f1_name]
        f2_state = fighter_state[f2_name]
        f1_state["elo"] = apply_elo_decay(f1_state["elo"], f1_state["last_fight_date"], fight["_parsed_date"])
        f2_state["elo"] = apply_elo_decay(f2_state["elo"], f2_state["last_fight_date"], fight["_parsed_date"])

        # Randomly assign fighter_a/fighter_b
        if random.random() < 0.5:
            a_name, b_name = f1_name, f2_name
            a_is_f1 = True
            b_is_f1 = False
        else:
            a_name, b_name = f2_name, f1_name
            a_is_f1 = False
            b_is_f1 = True

        # Compute pre-fight features
        feat_a = compute_stats(fighter_state[a_name], fight, a_is_f1, fighters_cache, category=fight["category"], priors=priors)
        feat_b = compute_stats(fighter_state[b_name], fight, b_is_f1, fighters_cache, category=fight["category"], priors=priors)

        if feat_a["is_debut"]:
            debut_a_count += 1
        if feat_b["is_debut"]:
            debut_b_count += 1
        if feat_a["stance"] == "Unknown":
            unknown_stance_count += 1
        if feat_b["stance"] == "Unknown":
            unknown_stance_count += 1

        # Height/reach from fights.json
        height_key_a = "fighter_1_height_cm" if a_is_f1 else "fighter_2_height_cm"
        reach_key_a = "fighter_1_reach_cm" if a_is_f1 else "fighter_2_reach_cm"

        height_a = fight.get(height_key_a)
        if height_a is None:
            height_a = fighters_cache.get(a_name, {}).get("height_cm")
        if height_a is None:
            height_a = np.nan
        else:
            height_a = float(height_a)

        reach_a = fight.get(reach_key_a)
        if reach_a is None:
            reach_a = fighters_cache.get(a_name, {}).get("reach_cm")
        if reach_a is None:
            reach_a = np.nan
        else:
            reach_a = float(reach_a)

        height_key_b = "fighter_1_height_cm" if b_is_f1 else "fighter_2_height_cm"
        reach_key_b = "fighter_1_reach_cm" if b_is_f1 else "fighter_2_reach_cm"

        height_b = fight.get(height_key_b)
        if height_b is None:
            height_b = fighters_cache.get(b_name, {}).get("height_cm")
        if height_b is None:
            height_b = np.nan
        else:
            height_b = float(height_b)

        reach_b = fight.get(reach_key_b)
        if reach_b is None:
            reach_b = fighters_cache.get(b_name, {}).get("reach_cm")
        if reach_b is None:
            reach_b = np.nan
        else:
            reach_b = float(reach_b)

        # Determine winner (1 = fighter_a, 2 = fighter_b)
        winner_name = fight["winner"]
        if winner_name == a_name:
            winner_label = 1
        elif winner_name == b_name:
            winner_label = 2
        elif winner_name == "Draw":
            winner_label = "Draw"
        elif winner_name == "No Contest":
            winner_label = "No Contest"
        else:
            winner_label = winner_name  # shouldn't happen, but fallback

        row = {
            "fight_id": f"fight_{idx:05d}",
            "event_date": fight["event_date"],
            "category": fight.get("category", ""),
            # title_bout intentionally omitted (buggy data)
            "fighter_a_name": a_name,
            "fighter_b_name": b_name,
            "age_a": feat_a["age"],
            "age_b": feat_b["age"],
            "age_diff": feat_a["age"] - feat_b["age"] if not (np.isnan(feat_a["age"]) or np.isnan(feat_b["age"])) else np.nan,
            "stance_a": feat_a["stance"],
            "stance_b": feat_b["stance"],
            "height_diff": height_a - height_b if not (np.isnan(height_a) or np.isnan(height_b)) else np.nan,
            "reach_diff": reach_a - reach_b if not (np.isnan(reach_a) or np.isnan(reach_b)) else np.nan,
            "win_pct_diff": feat_a["win_pct"] - feat_b["win_pct"] if not (np.isnan(feat_a["win_pct"]) or np.isnan(feat_b["win_pct"])) else np.nan,
            "ko_rate_diff": feat_a["ko_rate"] - feat_b["ko_rate"] if not (np.isnan(feat_a["ko_rate"]) or np.isnan(feat_b["ko_rate"])) else np.nan,
            "sub_rate_diff": feat_a["sub_rate"] - feat_b["sub_rate"] if not (np.isnan(feat_a["sub_rate"]) or np.isnan(feat_b["sub_rate"])) else np.nan,
            "dec_rate_diff": feat_a["dec_rate"] - feat_b["dec_rate"] if not (np.isnan(feat_a["dec_rate"]) or np.isnan(feat_b["dec_rate"])) else np.nan,
            "ko_loss_rate_diff": feat_a["ko_loss_rate"] - feat_b["ko_loss_rate"] if not (np.isnan(feat_a["ko_loss_rate"]) or np.isnan(feat_b["ko_loss_rate"])) else np.nan,
            "sub_loss_rate_diff": feat_a["sub_loss_rate"] - feat_b["sub_loss_rate"] if not (np.isnan(feat_a["sub_loss_rate"]) or np.isnan(feat_b["sub_loss_rate"])) else np.nan,
            "sig_str_landed_per_min_diff": feat_a["sig_str_landed_per_min"] - feat_b["sig_str_landed_per_min"] if not (np.isnan(feat_a["sig_str_landed_per_min"]) or np.isnan(feat_b["sig_str_landed_per_min"])) else np.nan,
            "sig_str_absorbed_per_min_diff": feat_a["sig_str_absorbed_per_min"] - feat_b["sig_str_absorbed_per_min"] if not (np.isnan(feat_a["sig_str_absorbed_per_min"]) or np.isnan(feat_b["sig_str_absorbed_per_min"])) else np.nan,
            "sig_str_accuracy_diff": feat_a["sig_str_accuracy"] - feat_b["sig_str_accuracy"] if not (np.isnan(feat_a["sig_str_accuracy"]) or np.isnan(feat_b["sig_str_accuracy"])) else np.nan,
            "td_avg_per_15min_diff": feat_a["td_avg_per_15min"] - feat_b["td_avg_per_15min"] if not (np.isnan(feat_a["td_avg_per_15min"]) or np.isnan(feat_b["td_avg_per_15min"])) else np.nan,
            "td_accuracy_diff": feat_a["td_accuracy"] - feat_b["td_accuracy"] if not (np.isnan(feat_a["td_accuracy"]) or np.isnan(feat_b["td_accuracy"])) else np.nan,
            "td_defense_diff": feat_a["td_defense"] - feat_b["td_defense"] if not (np.isnan(feat_a["td_defense"]) or np.isnan(feat_b["td_defense"])) else np.nan,
            "sub_att_per_15min_diff": feat_a["sub_att_per_15min"] - feat_b["sub_att_per_15min"] if not (np.isnan(feat_a["sub_att_per_15min"]) or np.isnan(feat_b["sub_att_per_15min"])) else np.nan,
            "ctrl_time_pct_diff": feat_a["ctrl_time_pct"] - feat_b["ctrl_time_pct"] if not (np.isnan(feat_a["ctrl_time_pct"]) or np.isnan(feat_b["ctrl_time_pct"])) else np.nan,
            "days_since_last_fight_diff": feat_a["days_since_last_fight"] - feat_b["days_since_last_fight"] if not (np.isnan(feat_a["days_since_last_fight"]) or np.isnan(feat_b["days_since_last_fight"])) else np.nan,
            "win_streak_diff": feat_a["current_win_streak"] - feat_b["current_win_streak"],
            "losing_streak_diff": feat_a["current_losing_streak"] - feat_b["current_losing_streak"],
            "total_fights_diff": feat_a["total_fights"] - feat_b["total_fights"],
            "elo_diff": feat_a["elo"] - feat_b["elo"],
            "is_debut_a": feat_a["is_debut"],
            "is_debut_b": feat_b["is_debut"],
            "recent_3_wins_a": feat_a["recent_3_wins"],
            "recent_3_wins_b": feat_b["recent_3_wins"],
            "recent_3_losses_a": feat_a["recent_3_losses"],
            "recent_3_losses_b": feat_b["recent_3_losses"],
            "recent_5_wins_a": feat_a["recent_5_wins"],
            "recent_5_wins_b": feat_b["recent_5_wins"],
            "recent_5_losses_a": feat_a["recent_5_losses"],
            "recent_5_losses_b": feat_b["recent_5_losses"],
            "recent_3_ko_loss_rate_a": feat_a["recent_3_ko_loss_rate"],
            "recent_3_ko_loss_rate_b": feat_b["recent_3_ko_loss_rate"],
            "recent_5_ko_loss_rate_a": feat_a["recent_5_ko_loss_rate"],
            "recent_5_ko_loss_rate_b": feat_b["recent_5_ko_loss_rate"],
            "decay_sig_per_min_a": feat_a["decay_sig_per_min"],
            "decay_sig_per_min_b": feat_b["decay_sig_per_min"],
            "decay_sig_absorbed_per_min_a": feat_a["decay_sig_absorbed_per_min"],
            "decay_sig_absorbed_per_min_b": feat_b["decay_sig_absorbed_per_min"],
            "decay_td_per_15min_a": feat_a["decay_td_per_15min"],
            "decay_td_per_15min_b": feat_b["decay_td_per_15min"],
            "avg_opp_elo_a": feat_a["avg_opp_elo"],
            "avg_opp_elo_b": feat_b["avg_opp_elo"],
            "avg_opp_elo_wins_a": feat_a["avg_opp_elo_wins"],
            "avg_opp_elo_wins_b": feat_b["avg_opp_elo_wins"],
            "recent_3_wins_diff": feat_a["recent_3_wins"] - feat_b["recent_3_wins"],
            "recent_3_losses_diff": feat_a["recent_3_losses"] - feat_b["recent_3_losses"],
            "recent_5_wins_diff": feat_a["recent_5_wins"] - feat_b["recent_5_wins"],
            "recent_5_losses_diff": feat_a["recent_5_losses"] - feat_b["recent_5_losses"],
            "recent_3_ko_loss_rate_diff": feat_a["recent_3_ko_loss_rate"] - feat_b["recent_3_ko_loss_rate"] if not (np.isnan(feat_a["recent_3_ko_loss_rate"]) or np.isnan(feat_b["recent_3_ko_loss_rate"])) else np.nan,
            "recent_5_ko_loss_rate_diff": feat_a["recent_5_ko_loss_rate"] - feat_b["recent_5_ko_loss_rate"] if not (np.isnan(feat_a["recent_5_ko_loss_rate"]) or np.isnan(feat_b["recent_5_ko_loss_rate"])) else np.nan,
            "decay_sig_per_min_diff": feat_a["decay_sig_per_min"] - feat_b["decay_sig_per_min"] if not (np.isnan(feat_a["decay_sig_per_min"]) or np.isnan(feat_b["decay_sig_per_min"])) else np.nan,
            "decay_sig_absorbed_per_min_diff": feat_a["decay_sig_absorbed_per_min"] - feat_b["decay_sig_absorbed_per_min"] if not (np.isnan(feat_a["decay_sig_absorbed_per_min"]) or np.isnan(feat_b["decay_sig_absorbed_per_min"])) else np.nan,
            "decay_td_per_15min_diff": feat_a["decay_td_per_15min"] - feat_b["decay_td_per_15min"] if not (np.isnan(feat_a["decay_td_per_15min"]) or np.isnan(feat_b["decay_td_per_15min"])) else np.nan,
            "avg_opp_elo_diff": feat_a["avg_opp_elo"] - feat_b["avg_opp_elo"] if not (np.isnan(feat_a["avg_opp_elo"]) or np.isnan(feat_b["avg_opp_elo"])) else np.nan,
            "avg_opp_elo_wins_diff": feat_a["avg_opp_elo_wins"] - feat_b["avg_opp_elo_wins"] if not (np.isnan(feat_a["avg_opp_elo_wins"]) or np.isnan(feat_b["avg_opp_elo_wins"])) else np.nan,
            "winner": winner_label,
        }
        rows.append(row)

        # --- Post-fight: update state ---
        # Determine outcome for both original fighters
        is_win_loss, win_side, finish_type = classify_method(
            fight["method"], fight["winner"], fight["fighter_1"], fight["fighter_2"]
        )

        f1_elo_before = fighter_state[fight["fighter_1"]]["elo"]
        f2_elo_before = fighter_state[fight["fighter_2"]]["elo"]

        update_state(fighter_state[fight["fighter_1"]], fight, True, is_win_loss, win_side, finish_type, opponent_elo=f2_elo_before)
        update_state(fighter_state[fight["fighter_2"]], fight, False, is_win_loss, win_side, finish_type, opponent_elo=f1_elo_before)

        # Elo update (with variable K-factor)
        f1_state = fighter_state[fight["fighter_1"]]
        f2_state = fighter_state[fight["fighter_2"]]
        if is_win_loss:
            if win_side == 1:
                score_a = 1.0
            else:
                score_a = 0.0
            k_a = get_k_factor(f1_state["total_fights"])
            k_b = get_k_factor(f2_state["total_fights"])
            f1_new_elo, f2_new_elo = elo_update(f1_state["elo"], f2_state["elo"], score_a, k_a=k_a, k_b=k_b)
            f1_state["elo"] = f1_new_elo
            f2_state["elo"] = f2_new_elo
        # else: no change for draws/no-contests

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    # Logging
    print(f"\nFinal rows generated: {len(df)}")
    print(f"Fights with is_debut_a=True: {debut_a_count}")
    print(f"Fights with is_debut_b=True: {debut_b_count}")
    print(f"Stance 'Unknown' entries: {unknown_stance_count}")
    print(f"\nNull summary per column:")
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df)) * 100
    null_df = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
    print(null_df[null_df["null_count"] > 0].to_string())
    print(f"\nDataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
