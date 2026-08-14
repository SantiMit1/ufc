"""Shared fighter-state, Elo and feature computation for all prediction scripts.

This module is the single source of truth for logic that was previously
copy-pasted across ``predict.py``, ``predict_event.py``, ``predict_batch.py``
and ``feature_engineering.py``:

- fight time parsing / total seconds (``parse_time``, ``fight_seconds``)
- finish classification (``classify_method``)
- Elo rating helpers (``get_k_factor``, ``apply_elo_decay``, ``elo_expected``, ``elo_update``)
- cumulative fighter state (``make_initial_state``, ``update_state``)
- per-fighter feature computation (``compute_stats_from_state``)
- chronological fighter histories (``build_fighter_states``)
- prediction rows / encoded feature frames (``build_prediction_row``)
- order-averaged fight prediction (``predict_fight``)
"""
import numpy as np
import pandas as pd
from datetime import datetime

from config import CUTOFF_DATE, ELO_K, ELO_INITIAL
from stats_utils import (
    shrink_rate,
    shrink_proportion,
    compute_composite_features,
    safe_int,
)


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


def classify_method(method: str, winner: str, fighter_1: str, fighter_2: str) -> tuple[bool, int, str | None]:
    """Return (is_win_loss, win_side, finish_type) for a fight.

    win_side is 1 if fighter_1 wins, 2 if fighter_2 wins, 0 otherwise.
    finish_type is 'KO', 'SUB', 'DEC' or None when there is no winner.
    """
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
        return 144.0
    elif total_fights <= 10:
        return 96.0
    elif total_fights <= 20:
        return 60.0
    else:
        return 36.0


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


def safe_sub(a: float, b: float) -> float:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (np.isnan(a) or np.isnan(b)):
            return a - b
    return np.nan


FEATURE_DIFF_FIELDS = [
    ("age", "age_diff"),
    ("win_pct", "win_pct_diff"),
    ("ko_rate", "ko_rate_diff"),
    ("sub_rate", "sub_rate_diff"),
    ("dec_rate", "dec_rate_diff"),
    ("ko_loss_rate", "ko_loss_rate_diff"),
    ("sub_loss_rate", "sub_loss_rate_diff"),
    ("sig_str_landed_per_min", "sig_str_landed_per_min_diff"),
    ("sig_str_absorbed_per_min", "sig_str_absorbed_per_min_diff"),
    ("sig_str_accuracy", "sig_str_accuracy_diff"),
    ("td_avg_per_15min", "td_avg_per_15min_diff"),
    ("td_accuracy", "td_accuracy_diff"),
    ("td_defense", "td_defense_diff"),
    ("sub_att_per_15min", "sub_att_per_15min_diff"),
    ("ctrl_time_pct", "ctrl_time_pct_diff"),
    ("days_since_last_fight", "days_since_last_fight_diff"),
    ("current_win_streak", "win_streak_diff"),
    ("current_losing_streak", "losing_streak_diff"),
    ("total_fights", "total_fights_diff"),
    ("elo", "elo_diff"),
    ("recent_3_wins", "recent_3_wins_diff"),
    ("recent_3_losses", "recent_3_losses_diff"),
    ("recent_5_wins", "recent_5_wins_diff"),
    ("recent_5_losses", "recent_5_losses_diff"),
    ("recent_3_ko_loss_rate", "recent_3_ko_loss_rate_diff"),
    ("recent_5_ko_loss_rate", "recent_5_ko_loss_rate_diff"),
    ("decay_sig_per_min", "decay_sig_per_min_diff"),
    ("decay_sig_absorbed_per_min", "decay_sig_absorbed_per_min_diff"),
    ("decay_td_per_15min", "decay_td_per_15min_diff"),
    ("avg_opp_elo", "avg_opp_elo_diff"),
    ("avg_opp_elo_wins", "avg_opp_elo_wins_diff"),
    ("striking", "striking_strength_diff"),
    ("grappling", "grappling_strength_diff"),
    ("durability", "durability_diff"),
    ("momentum", "momentum_diff"),
    ("experience", "experience_diff"),
]


def compute_feature_diffs(feat_a: dict, feat_b: dict) -> dict:
    """Compute all feature ``*_diff`` columns between two per-fighter feature dicts."""
    return {col: safe_sub(feat_a[key], feat_b[key]) for key, col in FEATURE_DIFF_FIELDS}


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
                             fighters_cache: dict, as_of_date: datetime,
                             category: str = "", priors: dict | None = None,
                             fight: dict | None = None,
                             is_fighter_1: bool | None = None) -> dict:
    """Compute per-fighter features from accumulated state (pre-fight).

    ``as_of_date`` is the reference date (event date for historical
    processing, or "now" for hypothetical fights). When ``fight`` and
    ``is_fighter_1`` are provided, the fighter's age is taken from the
    fight's scraped age fields; otherwise it is derived from the DOB in the
    fighters cache.
    """
    f = fighter_state
    total_fights = f["total_fights"]
    wins = f["wins"]
    losses = f["losses"]

    if fight is not None and is_fighter_1 is not None:
        age_key = "fighter_1_age" if is_fighter_1 else "fighter_2_age"
        age = fight.get(age_key)
        if age is None:
            age = np.nan
    else:
        entry = fighters_cache.get(fighter_name, {})
        dob_str = entry.get("dob")
        if dob_str:
            age = (as_of_date - datetime.strptime(dob_str, "%Y-%m-%d")).days / 365.25
        else:
            age = np.nan

    stance = get_stance(fighter_name, fighters_cache)
    win_pct = wins / total_fights if total_fights > 0 else np.nan
    ko_rate = f["wins_by_ko"] / wins if wins > 0 else 0.0
    sub_rate = f["wins_by_sub"] / wins if wins > 0 else 0.0
    dec_rate = f["wins_by_dec"] / wins if wins > 0 else 0.0
    ko_loss_rate = f["losses_by_ko"] / losses if losses > 0 else 0.0
    sub_loss_rate = f["losses_by_sub"] / losses if losses > 0 else 0.0

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
        days_since_last_fight = (as_of_date - f["last_fight_date"]).days

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
    w_sig = 0.0
    w_sig_abs = 0.0
    w_td = 0.0
    w_sec = 0.0
    for r in recent:
        years_ago = (as_of_date - r["date"]).days / 365.25
        w = np.exp(-LAMBDA * years_ago)
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


def _step_fight_state(fighter_state: dict, fight: dict, f1: str, f2: str) -> None:
    """Apply one fight's outcome to the shared state (update_state + Elo update)."""
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


class FightStateEngine:
    """Chronological fighter-state engine (single source of the Elo/state loop).

    Iterating yields each fight in chronological order. Before each yield the
    engine initializes both fighters' states and applies Elo decay so that
    ``.state`` only reflects fights strictly before the current one (no
    lookahead). When the consumer resumes the generator, the fight's outcome is
    applied (``update_state`` + Elo update) ready for the next iteration.

    Consumers read ``.state`` for pre-fight features/predictions, and must not
    mutate it before the engine's own post-fight update.
    """

    def __init__(self, fights: list, cutoff: datetime = CUTOFF_DATE):
        for fight in fights:
            fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")
        self.filtered = sorted(
            [f for f in fights if f["_parsed_date"] >= cutoff],
            key=lambda f: f["_parsed_date"],
        )
        self.state: dict[str, dict] = {}

    def __iter__(self):
        for fight in self.filtered:
            f1, f2 = fight["fighter_1"], fight["fighter_2"]
            if f1 not in self.state:
                self.state[f1] = make_initial_state()
            if f2 not in self.state:
                self.state[f2] = make_initial_state()

            # Apply Elo decay for inactivity (>1 year)
            self.state[f1]["elo"] = apply_elo_decay(
                self.state[f1]["elo"], self.state[f1]["last_fight_date"], fight["_parsed_date"])
            self.state[f2]["elo"] = apply_elo_decay(
                self.state[f2]["elo"], self.state[f2]["last_fight_date"], fight["_parsed_date"])

            yield fight
            _step_fight_state(self.state, fight, f1, f2)


def build_fighter_states(fights: list, fighters_cache: dict) -> dict:
    engine = FightStateEngine(fights)
    for _ in engine:
        pass
    return engine.state


def build_prediction_row(f1: str, f2: str, fighter_states: dict,
                         fighters_cache: dict, current_date: datetime,
                         category: str, priors: dict | None,
                         feature_meta: dict) -> pd.DataFrame:
    """Build the encoded feature frame (X) for a fight with order (f1, f2)."""
    def get_phys(name, key):
        v = fighters_cache.get(name, {}).get(key)
        return float(v) if v is not None else np.nan

    height1, reach1 = get_phys(f1, "height_cm"), get_phys(f1, "reach_cm")
    height2, reach2 = get_phys(f2, "height_cm"), get_phys(f2, "reach_cm")

    state1 = fighter_states.get(f1, make_initial_state())
    state2 = fighter_states.get(f2, make_initial_state())

    feat1 = compute_stats_from_state(state1, f1, fighters_cache, current_date,
                                     category=category, priors=priors)
    feat2 = compute_stats_from_state(state2, f2, fighters_cache, current_date,
                                     category=category, priors=priors)

    row = {}
    row["age_a"] = feat1["age"]
    row["age_b"] = feat2["age"]
    row["stance_a"] = feat1["stance"]
    row["stance_b"] = feat2["stance"]
    row["category"] = category
    row["height_diff"] = safe_sub(height1, height2)
    row["reach_diff"] = safe_sub(reach1, reach2)
    row.update(compute_feature_diffs(feat1, feat2))

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
    return X_encoded


def predict_fight(fighter_a: str, fighter_b: str, category: str,
                  fighter_states: dict, fighters_cache: dict,
                  model, feature_meta: dict, current_date: datetime,
                  priors: dict | None = None) -> tuple:
    """Predict a single fight and return (prob_a, prob_b) raw probabilities.

    Predicts in both orderings and averages to remove order-dependent bias.
    Callers are responsible for rounding/formatting for display.
    """
    prob_a_forward = model.predict_proba(
        build_prediction_row(fighter_a, fighter_b, fighter_states, fighters_cache,
                             current_date, category, priors, feature_meta)
    )[0, 1]
    prob_b_forward = model.predict_proba(
        build_prediction_row(fighter_b, fighter_a, fighter_states, fighters_cache,
                             current_date, category, priors, feature_meta)
    )[0, 1]
    prob_a = (prob_a_forward + (1.0 - prob_b_forward)) / 2.0
    return prob_a, 1.0 - prob_a
