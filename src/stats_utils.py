import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV

from config import CUTOFF_DATE, FIGHTERS_CACHE_PATH, FIGHTS_PATH

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


# ── Composite feature groups (semantic grouping, no hardcoded weights) ──
COMPOSITE_GROUPS = {
    "striking": [
        "sig_str_landed_per_min",
        "sig_str_absorbed_per_min",
        "sig_str_accuracy",
        "decay_sig_per_min",
        "decay_sig_absorbed_per_min",
        "ko_rate",
        "dec_rate",
    ],
    "grappling": [
        "td_avg_per_15min",
        "td_accuracy",
        "td_defense",
        "sub_att_per_15min",
        "ctrl_time_pct",
        "sub_rate",
        "decay_td_per_15min",
    ],
    "durability": [
        "ko_loss_rate",
        "sub_loss_rate",
        "recent_3_ko_loss_rate",
        "recent_5_ko_loss_rate",
    ],
    "momentum": [
        "win_pct",
        "recent_3_wins",
        "recent_3_losses",
        "recent_5_wins",
        "recent_5_losses",
        "current_win_streak",
        "current_losing_streak",
        "days_since_last_fight",
    ],
    "experience": [
        "avg_opp_elo",
        "avg_opp_elo_wins",
        "total_fights",
    ],
}

# Hardcoded domain signs for interpretable composites (positive = good for fighter)
COMPOSITE_SIGNS = {
    "striking": {
        "sig_str_landed_per_min": 1,
        "sig_str_absorbed_per_min": -1,
        "sig_str_accuracy": 1,
        "decay_sig_per_min": 1,
        "decay_sig_absorbed_per_min": -1,
        "ko_rate": 1,
        "dec_rate": 1,
    },
    "grappling": {
        "td_avg_per_15min": 1,
        "td_accuracy": 1,
        "td_defense": 1,
        "sub_att_per_15min": 1,
        "ctrl_time_pct": 1,
        "sub_rate": 1,
        "decay_td_per_15min": 1,
    },
    "durability": {
        "ko_loss_rate": -1,
        "sub_loss_rate": -1,
        "recent_3_ko_loss_rate": -1,
        "recent_5_ko_loss_rate": -1,
    },
    "momentum": {
        "win_pct": 1,
        "recent_3_wins": 1,
        "recent_3_losses": -1,
        "recent_5_wins": 1,
        "recent_5_losses": -1,
        "current_win_streak": 1,
        "current_losing_streak": -1,
        "days_since_last_fight": -1,
    },
    "experience": {
        "avg_opp_elo": 1,
        "avg_opp_elo_wins": 1,
        "total_fights": 1,
    },
}

def fit_composite_params(df_diffs: "pd.DataFrame", y: "np.ndarray | pd.Series", method: str = "gain",
                         random_state: int = 42) -> dict:
    """Fit dynamic composite params from training diff data.

    Args:
        df_diffs: DataFrame with raw diff columns (one row per fight, diff = A-B).
                  Expected cols are all features in COMPOSITE_GROUPS.
        y: binary target (1 if A wins).
        method: "gain" | "equal" | "ridge"
        random_state: seed for probe model.

    Returns:
        dict with keys: means, stds, weights, method
        - means[g][f]: mean of diff col f in train (for centering)
        - stds[g][f]: std of diff col f in train (for scaling, floored at 1e-9)
        - weights[g][f]: signed weight magnitude normalized per group (L1 sum = 1 per group)
    """

    y_arr = np.asarray(y).ravel()
    # For per-fighter composites we only need std (scale); means are 0 so
    # composite_diff = sum(raw_diff/std * w) = composite_a - composite_b.
    # Diff means are ~0, so centering has negligible effect; 0 keeps stability.
    means: dict = {}
    stds: dict = {}
    for g, feats in COMPOSITE_GROUPS.items():
        means[g] = {}
        stds[g] = {}
        for f in feats:
            col = f + "_diff"
            if col in df_diffs.columns:
                vals = pd.to_numeric(df_diffs[col], errors="coerce")
                s = float(np.nanstd(vals.values)) if len(vals) > 0 else 1.0
                if not np.isfinite(s) or s < 1e-9:
                    s = 1.0
                means[g][f] = 0.0
                stds[g][f] = s
            else:
                means[g][f] = 0.0
                stds[g][f] = 1.0

    weights: dict = {}
    if method == "equal":
        for g, feats in COMPOSITE_GROUPS.items():
            signs = COMPOSITE_SIGNS.get(g, {})
            # uniform magnitude within group
            mag = 1.0 / max(len(feats), 1)
            weights[g] = {f: float(signs.get(f, 1) * mag) for f in feats}

    elif method == "ridge":
        for g, feats in COMPOSITE_GROUPS.items():
            cols = [f + "_diff" for f in feats if f + "_diff" in df_diffs.columns]
            if not cols:
                weights[g] = {f: float(COMPOSITE_SIGNS.get(g, {}).get(f, 1) * (1.0 / len(feats))) for f in feats}
                continue
            Xg = df_diffs[cols].copy()
            # median impute NaNs
            imp = SimpleImputer(strategy="median")
            Xg_imp = imp.fit_transform(Xg)
            # standardize by std for ridge, then fit
            # Use precomputed stds for consistency
            std_vec = np.array([stds[g][f] for f in feats if f + "_diff" in df_diffs.columns], dtype=float)
            std_vec = np.where(std_vec < 1e-9, 1.0, std_vec)
            Xg_std = Xg_imp / std_vec
            alphas = [0.1, 1.0, 10.0, 100.0]
            ridge = RidgeCV(alphas=alphas, cv=3)
            try:
                ridge.fit(Xg_std, y_arr)
                coefs = ridge.coef_
            except Exception:
                coefs = np.zeros(len(cols))
            # Map coefs back to feature names, enforce domain sign if coef has opposite sign and large magnitude?
            # Keep ridge sign as learned, but clip near-zero
            w_dict = {}
            for f, c in zip(feats, coefs):
                # coefs correspond to standardized features; use coef directly
                # If coef is NaN, fallback to equal
                if not np.isfinite(c):
                    c = 0.0
                w_dict[f] = float(c)
            # If all coefs ~0, fallback to equal
            if all(abs(v) < 1e-9 for v in w_dict.values()):
                mag = 1.0 / len(feats)
                w_dict = {f: float(COMPOSITE_SIGNS.get(g, {}).get(f, 1) * mag) for f in feats}
            else:
                # L1 normalize per group to keep scale comparable across methods
                l1 = sum(abs(v) for v in w_dict.values())
                if l1 > 1e-9:
                    w_dict = {k: v / l1 for k, v in w_dict.items()}
            weights[g] = w_dict

    elif method == "gain":
        # Build probe dataset with available raw diffs
        all_feats = [f for feats in COMPOSITE_GROUPS.values() for f in feats]
        cols = [f + "_diff" for f in all_feats if f + "_diff" in df_diffs.columns]
        if not cols:
            raise ValueError("No raw diff columns found for gain fit")
        Xp = df_diffs[cols].copy()
        # LGBM handles NaN natively, no impute needed
        probe = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, verbose=-1, random_state=random_state, n_jobs=-1
        )
        # Use simple train/val split for early stopping to avoid overfit
        n = len(Xp)
        # chronological split: last 15% as val for early stopping if n large
        if n > 200:
            split = int(n * 0.85)
            Xtr, Xva = Xp.iloc[:split], Xp.iloc[split:]
            ytr, yva = y_arr[:split], y_arr[split:]
            probe.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        else:
            probe.fit(Xp, y_arr)
        # Gain importance
        booster = probe.booster_
        gains = booster.feature_importance(importance_type="gain")
        feat_names = booster.feature_name()
        gain_map = {name: float(g) for name, g in zip(feat_names, gains)}
        # For features not in booster (all NaN cols), gain 0
        for col in cols:
            if col not in gain_map:
                gain_map[col] = 0.0
        for g, feats in COMPOSITE_GROUPS.items():
            w_dict = {}
            signs = COMPOSITE_SIGNS.get(g, {})
            for f in feats:
                col = f + "_diff"
                ge = gain_map.get(col, 0.0)
                sign = signs.get(f, 1)
                # Use gain magnitude * sign
                w_dict[f] = float(ge * sign)
            # L1 normalize per group
            l1 = sum(abs(v) for v in w_dict.values())
            if l1 < 1e-9:
                # fallback to equal if no gain (e.g., all zeros early fights)
                mag = 1.0 / len(feats)
                w_dict = {f: float(signs.get(f, 1) * mag) for f in feats}
            else:
                w_dict = {k: v / l1 for k, v in w_dict.items()}
            weights[g] = w_dict
    else:
        raise ValueError(f"Unknown composite method: {method}")

    return {"means": means, "stds": stds, "weights": weights, "method": method, "version": 1}


def compute_composite_features(feat: dict, composite_params: dict) -> dict:
    """Compute 5 composite features from per-fighter feat dict using dynamic params."""
    means = composite_params.get("means", {})
    stds = composite_params.get("stds", {})
    weights = composite_params.get("weights", {})
    composites = {}
    for g in COMPOSITE_GROUPS:
        wdict = weights.get(g, {})
        m_dict = means.get(g, {})
        s_dict = stds.get(g, {})
        value = 0.0
        has_any = False
        for f, w in wdict.items():
            v = feat.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            m = m_dict.get(f, 0.0)
            s = s_dict.get(f, 1.0)
            if s is None or s == 0 or (isinstance(s, float) and np.isnan(s)):
                s = 1.0
            # z-score: (v - mean) / std, then weighted sum
            z = (v - m) / s
            value += z * w
            has_any = True
        composites[g] = value if has_any else np.nan
    return composites


def recompute_composite_diffs(df: "pd.DataFrame", composite_params: dict) -> "pd.DataFrame":
    """Recompute composite *_diff cols in a dataset DataFrame from raw diffs using params.

    Useful in train_model to replace stub csv values with fitted composites.
    Operates on a copy and returns it.
    """

    out = df.copy()
    # For each composite, recompute diff as weighted sum of standardized raw diffs: sum(raw_diff/std * w)
    # (means are 0 by design, so no centering)
    stds = composite_params.get("stds", {})
    weights = composite_params.get("weights", {})
    for g in COMPOSITE_GROUPS:
        wdict = weights.get(g, {})
        s_dict = stds.get(g, {})
        vals = None
        for f, w in wdict.items():
            col = f + "_diff"
            if col not in out.columns:
                continue
            s = s_dict.get(f, 1.0)
            if s == 0 or (isinstance(s, float) and np.isnan(s)):
                s = 1.0
            z = pd.to_numeric(out[col], errors="coerce") / s
            contrib = z * w
            if vals is None:
                vals = contrib
            else:
                vals = vals.add(contrib, fill_value=0)
        if vals is not None:
            col_map = {
                "striking": "striking_strength_diff",
                "grappling": "grappling_strength_diff",
                "durability": "durability_diff",
                "momentum": "momentum_diff",
                "experience": "experience_diff",
            }
            target_col = col_map[g]
            out[target_col] = vals
    return out
