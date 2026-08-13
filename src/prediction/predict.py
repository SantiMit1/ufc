import sys
import json
import numpy as np
import joblib
import shap
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    FIGHTS_PATH, FIGHTERS_CACHE_PATH, MODEL_PATH, FEATURE_COLS_PATH,
    CUTOFF_DATE, WEIGHT_CLASSES,
)
from fighter_engine import (
    make_initial_state, compute_stats_from_state, build_fighter_states,
    build_prediction_row,
)
from stats_utils import PriorAccumulator




























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
    prior_accum = PriorAccumulator()
    # Parse dates first
    for fight in fights:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    # Sort and add to accumulator
    for fight in sorted(fights, key=lambda f: f["_parsed_date"]):
        if fight["_parsed_date"] >= CUTOFF_DATE:
            prior_accum.add(fight)
    priors = prior_accum.priors()
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
    print("\n  Weight class:")
    for i, wc in enumerate(WEIGHT_CLASSES, 1):
        print(f"    {i}. {wc}")
    wc_input = input("\n  Select (number or name): ").strip()
    try:
        category = WEIGHT_CLASSES[int(wc_input) - 1]
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
        X_encoded = build_prediction_row(
            f1, f2, fighter_states, fighters_cache, current_date,
            category, priors, feature_meta,
        )
        prob = model.predict_proba(X_encoded)[0, 1]

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
        return prob, shap_vals

    # Predict in both orders and average to remove order-dependent bias
    prob_a_forward, shap_a_forward = predict_order(fighter_a, fighter_b)
    prob_b_forward, shap_b_forward = predict_order(fighter_b, fighter_a)
    prob_a = (prob_a_forward + (1.0 - prob_b_forward)) / 2.0
    prob_b = 1.0 - prob_a

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
    print(f"\n  Favorite: {favorite} ({fav_prob * 100:.1f}%)")
    print(f"  Underdog: {underdog} ({dog_prob * 100:.1f}%)")

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




if __name__ == "__main__":
    main()
