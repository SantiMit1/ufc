import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any

from config import FIGHTS_PATH, FIGHTERS_CACHE_PATH, CUTOFF_DATE

PRIOR_MINUTES = 10.0
PRIOR_ATTEMPTS = 10
MIN_CATEGORY_FIGHTS = 200
SHRINK_FIGHTS_CUTOFF = 5


def safe_int(val: Any) -> int:
    return int(val) if val is not None else 0


def _fight_minutes(fight: dict) -> float:
    if fight.get("round", 0) == 0:
        return 0.0
    time_str = fight.get("time", "") or ""
    parts = time_str.split(":")
    minutes = int(parts[0]) if parts else 0
    seconds = int(parts[1]) if len(parts) > 1 else 0
    minutes_this_round = minutes + seconds / 60.0
    completed = (fight["round"] - 1) * 5
    return float(completed + minutes_this_round) if completed + minutes_this_round > 0 else 0.0


def _effective_strength(total_fights: int, base_strength: float) -> float:
    if total_fights >= SHRINK_FIGHTS_CUTOFF:
        return 1.0
    frac = 1.0 - total_fights / SHRINK_FIGHTS_CUTOFF
    return max(1.0, base_strength * frac)


def shrink_rate(
    observed: float,
    denominator: float,
    prior_mean: float,
    prior_strength: float = PRIOR_MINUTES,
    total_fights: int = 0,
) -> float:
    if denominator is None or denominator <= 0 or np.isnan(denominator):
        return prior_mean
    if np.isnan(observed) or observed is None:
        return prior_mean
    strength = _effective_strength(total_fights, prior_strength)
    return (observed + prior_mean * strength) / (denominator + strength)


def shrink_proportion(
    successes: int,
    attempts: int,
    prior_mean: float,
    prior_strength: int = PRIOR_ATTEMPTS,
    total_fights: int = 0,
) -> float:
    if attempts is None or attempts <= 0:
        return prior_mean
    strength = _effective_strength(total_fights, float(prior_strength))
    return (successes + prior_mean * strength) / (attempts + strength)




def _accumulate_fight(accum: dict, fight: dict) -> None:
    """Add a single fight's stats to a prior accumulator dict."""
    cat = fight.get("category", "").strip()
    if not cat:
        cat = "global"
    if cat not in accum:
        accum[cat] = {
            "count": 0,
            "sig_landed": 0,
            "sig_attempted": 0,
            "sig_absorbed": 0,
            "td_landed": 0,
            "td_attempted": 0,
            "td_against_landed": 0,
            "td_against_attempted": 0,
            "minutes": 0.0,
        }

    minutes = _fight_minutes(fight)

    for side in ("stats_fighter_1", "stats_fighter_2"):
        stats = fight.get(side, {})
        opp_side = "stats_fighter_2" if side == "stats_fighter_1" else "stats_fighter_1"
        opp_stats = fight.get(opp_side, {})

        sig_landed = safe_int(stats.get("sig_strikes", {}).get("landed"))
        sig_attempted = safe_int(stats.get("sig_strikes", {}).get("attempted"))
        sig_absorbed = safe_int(opp_stats.get("sig_strikes", {}).get("landed"))
        td_landed = safe_int(stats.get("takedowns", {}).get("landed"))
        td_attempted = safe_int(stats.get("takedowns", {}).get("attempted"))
        td_against_landed = safe_int(opp_stats.get("takedowns", {}).get("landed"))
        td_against_attempted = safe_int(opp_stats.get("takedowns", {}).get("attempted"))

        accum[cat]["count"] += 1
        accum[cat]["sig_landed"] += sig_landed
        accum[cat]["sig_attempted"] += sig_attempted
        accum[cat]["sig_absorbed"] += sig_absorbed
        accum[cat]["td_landed"] += td_landed
        accum[cat]["td_attempted"] += td_attempted
        accum[cat]["td_against_landed"] += td_against_landed
        accum[cat]["td_against_attempted"] += td_against_attempted
        accum[cat]["minutes"] += minutes


def _build_priors_from_accum(accum: dict) -> dict:
    """Build per-category priors (with global fallback) from an accumulator dict."""
    priors = {}
    for cat, a in accum.items():
        if a["count"] < MIN_CATEGORY_FIGHTS and cat != "global":
            continue
        minutes = a["minutes"]
        priors[cat] = {
            "sig_str_landed_per_min": a["sig_landed"] / minutes if minutes > 0 else 0.0,
            "sig_str_absorbed_per_min": a["sig_absorbed"] / minutes if minutes > 0 else 0.0,
            "sig_str_accuracy": a["sig_landed"] / a["sig_attempted"] if a["sig_attempted"] > 0 else 0.0,
            "td_avg_per_15min": a["td_landed"] / minutes * 15.0 if minutes > 0 else 0.0,
            "td_accuracy": a["td_landed"] / a["td_attempted"] if a["td_attempted"] > 0 else 0.0,
            "td_defense": 1.0 - a["td_against_landed"] / a["td_against_attempted"] if a["td_against_attempted"] > 0 else 0.0,
        }

    # Ensure global exists
    if "global" not in priors and accum:
        total_min = sum(a["minutes"] for a in accum.values())
        total_sig_l = sum(a["sig_landed"] for a in accum.values())
        total_sig_a = sum(a["sig_attempted"] for a in accum.values())
        total_sig_ab = sum(a["sig_absorbed"] for a in accum.values())
        total_td_l = sum(a["td_landed"] for a in accum.values())
        total_td_a = sum(a["td_attempted"] for a in accum.values())
        total_td_al = sum(a["td_against_landed"] for a in accum.values())
        total_td_aa = sum(a["td_against_attempted"] for a in accum.values())
        priors["global"] = {
            "sig_str_landed_per_min": total_sig_l / total_min if total_min > 0 else 0.0,
            "sig_str_absorbed_per_min": total_sig_ab / total_min if total_min > 0 else 0.0,
            "sig_str_accuracy": total_sig_l / total_sig_a if total_sig_a > 0 else 0.0,
            "td_avg_per_15min": total_td_l / total_min * 15.0 if total_min > 0 else 0.0,
            "td_accuracy": total_td_l / total_td_a if total_td_a > 0 else 0.0,
            "td_defense": 1.0 - total_td_al / total_td_aa if total_td_aa > 0 else 0.0,
        }

    # Fill missing categories with global
    for cat in list(accum.keys()):
        if cat not in priors:
            priors[cat] = dict(priors.get("global", {}))

    return priors


class PriorAccumulator:
    """Incremental (no-lookahead) population-prior accumulator.

    Add fights in chronological order. To predict a fight, read ``priors()``
    BEFORE ``add()``-ing that fight so the priors only reflect earlier ones.
    """

    def __init__(self):
        self._accum: dict = {}

    def add(self, fight: dict) -> None:
        """Add a single fight's stats to the accumulator."""
        _accumulate_fight(self._accum, fight)

    def priors(self) -> dict:
        """Build per-category priors (with global fallback) from accumulated fights."""
        return _build_priors_from_accum(self._accum)


# ── Shared data loading (all scripts; run from repo root) ──
def load_fights(path: Path = FIGHTS_PATH) -> list:
    """Load the fights dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fighter_cache(path: Path = FIGHTERS_CACHE_PATH) -> dict:
    """Load the fighters info cache from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_fights(fights: list, cutoff: datetime = CUTOFF_DATE) -> list:
    """Set ``_parsed_date`` on each fight and return the chronological, cutoff-filtered list."""
    for fight in fights:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    return sorted(
        [f for f in fights if f["_parsed_date"] >= cutoff],
        key=lambda f: f["_parsed_date"],
    )


def compute_priors(fights: list, cutoff: datetime | None = CUTOFF_DATE) -> dict:
    """Accumulate all fights (chronological, optional cutoff) and build the prior dict.

    Pass ``cutoff=None`` to include fights before ``CUTOFF_DATE`` (used for
    in-time event simulation, which historically did not apply the cutoff).
    """
    acc = PriorAccumulator()
    for fight in sorted(fights, key=lambda f: f.get("event_date", "")):
        d = datetime.strptime(fight["event_date"], "%Y-%m-%d")
        if cutoff is None or d >= cutoff:
            acc.add(fight)
    return acc.priors()


# ── Composite feature weights (from LightGBM gain importance × domain sign) ──
COMPOSITE_WEIGHTS = {
    "striking": {
        "sig_str_landed_per_min": 0.156,
        "sig_str_absorbed_per_min": -0.220,
        "sig_str_accuracy": 0.177,
        "decay_sig_per_min": 0.099,
        "decay_sig_absorbed_per_min": -0.220,
        "ko_rate": 0.050,
        "dec_rate": 0.078,
    },
    "grappling": {
        "td_avg_per_15min": 0.089,
        "td_accuracy": 0.194,
        "td_defense": 0.120,
        "sub_att_per_15min": 0.126,
        "ctrl_time_pct": 0.099,
        "sub_rate": 0.089,
        "decay_td_per_15min": 0.283,
    },
    "durability": {
        "ko_loss_rate": -0.351,
        "sub_loss_rate": -0.243,
        "recent_3_ko_loss_rate": -0.135,
        "recent_5_ko_loss_rate": -0.270,
    },
    "momentum": {
        "win_pct": 0.064,
        "recent_3_wins": 0.048,
        "recent_3_losses": -0.136,
        "recent_5_wins": 0.072,
        "recent_5_losses": -0.040,
        "current_win_streak": 0.128,
        "current_losing_streak": -0.032,
        "days_since_last_fight": -0.280,
    },
    "experience": {
        "avg_opp_elo": 0.28,
        "avg_opp_elo_wins": 0.36,
        "total_fights": 0.36,
    },
}

# Feature scale (std) for fixed-scale normalization so weights are comparable
# across features of very different magnitudes (e.g. Elo-scale vs fight counts).
# Factor = 1 / std, so each sub-feature contributes weight * std-units.
COMPOSITE_SCALES = {
    "striking": {
        "sig_str_landed_per_min": 1.0 / 1.07,
        "sig_str_absorbed_per_min": 1.0 / 1.11,
        "sig_str_accuracy": 1.0 / 0.078,
        "decay_sig_per_min": 1.0 / 1.79,
        "decay_sig_absorbed_per_min": 1.0 / 1.96,
        "ko_rate": 1.0 / 0.299,
        "dec_rate": 1.0 / 0.316,
    },
    "grappling": {
        "td_avg_per_15min": 1.0 / 1.11,
        "td_accuracy": 1.0 / 0.122,
        "td_defense": 1.0 / 0.124,
        "sub_att_per_15min": 1.0 / 1.37,
        "ctrl_time_pct": 1.0 / 0.150,
        "sub_rate": 1.0 / 0.275,
        "decay_td_per_15min": 1.0 / 1.75,
    },
    "durability": {
        "ko_loss_rate": 1.0 / 0.338,
        "sub_loss_rate": 1.0 / 0.302,
        "recent_3_ko_loss_rate": 1.0 / 0.219,
        "recent_5_ko_loss_rate": 1.0 / 0.198,
    },
    "momentum": {
        "win_pct": 1.0 / 0.224,
        "recent_3_wins": 1.0 / 0.769,
        "recent_3_losses": 1.0 / 0.747,
        "recent_5_wins": 1.0 / 1.06,
        "recent_5_losses": 1.0 / 0.994,
        "current_win_streak": 1.0 / 1.33,
        "current_losing_streak": 1.0 / 0.699,
        "days_since_last_fight": 1.0 / 188.0,
    },
    "experience": {
        "avg_opp_elo": 1.0 / 38.0,   # Elo-scale
        "avg_opp_elo_wins": 1.0 / 38.0,
        "total_fights": 1.0 / 4.7,   # fight-count scale
    },
}


def compute_composite_features(feat: dict) -> dict:
    composites = {}
    for name, weights in COMPOSITE_WEIGHTS.items():
        scales = COMPOSITE_SCALES.get(name, {})
        value = 0.0
        has_any = False
        for subfeat, weight in weights.items():
            v = feat.get(subfeat)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                factor = scales.get(subfeat, 1.0)
                value += v * weight * factor
                has_any = True
        composites[name] = value if has_any else np.nan
    return composites
