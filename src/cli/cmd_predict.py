"""Predict commands: fight, event, url."""
import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

from cli._common import resolve_paths
from config import WEIGHT_CLASSES
from fighter_engine import (
    build_fighter_states,
    build_historical_context,
    build_prediction_row,
    compute_stats_from_state,
    is_debut,
    make_initial_state,
    predict_fight,
)
from stats_utils import compute_priors, load_fighter_cache, load_fights

try:
    import shap
except ImportError:
    shap = None

# --- case-insensitive helpers ---
def _canonical_weight_class(cat: str | None) -> str | None:
    if cat is None:
        return None
    s = cat.strip()
    if not s:
        return cat
    low = s.lower()
    # map case-insensitively to canonical WEIGHT_CLASSES
    mapping = {c.lower(): c for c in WEIGHT_CLASSES}
    # also handle common variants: "light weight" -> "lightweight", remove spaces/hyphens
    norm = low.replace(" ", "").replace("-", "").replace("_", "")
    for k, v in mapping.items():
        if k.replace(" ", "").replace("-", "") == norm:
            return v
    return mapping.get(low, s)  # fallback to original if not found (will warn)

def _canonical_fighter(name: str, cache: dict, states: dict | None = None) -> str:
    s = name.strip()
    if not s:
        return name
    # exact match first
    if s in cache or (states is not None and s in states):
        return s
    # build lower map
    all_names = set(cache.keys())
    if states is not None:
        all_names.update(states.keys())
    lower_map = {n.lower(): n for n in all_names}
    low = s.lower()
    if low in lower_map:
        return lower_map[low]
    # also try substring? For predict we want exact case-insensitive, not fuzzy,
    # but we can keep exact lower match only. Return original if not found.
    return lower_map.get(low, s)

# ---------- helpers ----------

def _load_data(args):
    fights_path, cache_path, _, model_path, features_path = resolve_paths(args)
    # load_fights/load_fighter_cache accept path param
    try:
        fights = load_fights(fights_path)
    except FileNotFoundError:
        print(f"error: fights not found at {fights_path}", file=sys.stderr)
        sys.exit(1)
    try:
        cache = load_fighter_cache(cache_path)
    except FileNotFoundError:
        print(f"error: fighters cache not found at {cache_path}", file=sys.stderr)
        sys.exit(1)
    return fights, cache, model_path, features_path


def _print_event_table(event_name, event_date, predictions, skipped):
    # predictions: list of dicts with fighter_a, fighter_b, prob_a, prob_b, category, winner
    print()
    print("=" * 100)
    print(f"  {event_name}  ({event_date})")
    print("=" * 100)
    if predictions:
        header = f"  {'#':<4} {'Fighter A':<26} {'Fighter B':<26} {'Category':<20} {'Prob A':<8} {'Prob B':<8} {'Winner':<12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, p in enumerate(predictions, 1):
            fa = p["fighter_a"][:26]
            fb = p["fighter_b"][:26]
            cat = p.get("category", "")[:20]
            pa = f"{p['prob_a']*100:5.1f}%"
            pb = f"{p['prob_b']*100:5.1f}%"
            winner = p.get("winner", "")[:12]
            err = p.get("error", "")
            if err:
                winner = f"ERR:{err[:10]}"
            print(f"  {i:<4} {fa:<26} {fb:<26} {cat:<20} {pa:<8} {pb:<8} {winner:<12}")
        print("=" * 100)
    if skipped:
        print(f"\n  SKIPPED ({len(skipped)}):")
        for s in skipped:
            print(f"    - {s['fighter_a']} vs {s['fighter_b']} ({s.get('category','')}): {s.get('reason','')}")

    print()


def _print_url_table(event_name, event_date, url, results, skipped):
    print()
    print("=" * 120)
    title = event_name or url
    print(f"  {title}  ({event_date.strftime('%B %d, %Y')})")
    print("=" * 120)
    header = (f"  {'#':<4s} {'Fighter A':<32s} {'Fighter B':<32s} {'Category':<20s} "
              f"{'Rnd':<4s} {'Prob A':<8s} {'Prob B':<8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, (f1, f2, cat, prob_a, prob_b, tf1, tf2, rounds, rec1, rec2) in enumerate(results, 1):
        r1 = f"({rec1[0]}-{rec1[1]}-{rec1[2]})"
        r2 = f"({rec2[0]}-{rec2[1]}-{rec2[2]})"
        n1 = f"{f1} {r1}{'*' if tf1 < 3 else ''}"
        n2 = f"{f2} {r2}{'*' if tf2 < 3 else ''}"
        rnd = "5" if rounds == 5 else "3"
        print(f"  {i:<4d} {n1:<32s} {n2:<32s} {cat:<20s} {rnd:<4s} "
              f"{prob_a*100:6.1f}% {prob_b*100:6.1f}%")
    print("=" * 120)
    low = [(f1, f2, tf1, tf2, rec1, rec2) for f1, f2, _, _, _, tf1, tf2, _, rec1, rec2 in results if tf1 < 3 or tf2 < 3]
    if low:
        print("\n  * Fighter has fewer than 3 UFC fights before this event — prediction may be unreliable.")
        for f1, f2, tf1, tf2, rec1, rec2 in low:
            r1 = f"({rec1[0]}-{rec1[1]}-{rec1[2]})"
            r2 = f"({rec2[0]}-{rec2[1]}-{rec2[2]})"
            print(f"    - {f1} {r1}{'*' if tf1 < 3 else ''} vs {f2} {r2}{'*' if tf2 < 3 else ''}")
    if skipped:
        print(f"\n  SKIPPED ({len(skipped)} fight(s)):")
        for f1, f2, cat, reason in skipped:
            print(f"    - {f1} vs {f2} ({cat}): {reason}")
    print()


# ---------- fight ----------

def handle_fight(args):
    fights, cache, model_path, features_path = _load_data(args)


    # Load model/meta
    model = joblib.load(model_path)
    feature_meta = joblib.load(features_path)
    model_type = feature_meta.get("model_type", "stacking")

    # Non-interactive mode
    if args.fighter_a and args.fighter_b:
        raw_a = args.fighter_a.strip()
        raw_b = args.fighter_b.strip()
        raw_cat = args.weight_class
        category = _canonical_weight_class(raw_cat) if raw_cat else "Lightweight"
        if not args.weight_class:
            print(f"warning: --weight-class not provided, defaulting to {category}", file=sys.stderr)
        max_rounds = args.rounds if args.rounds else 3

        # Build states/priors first to allow fighter canonicalization
        priors = compute_priors(fights)
        fighter_states = build_fighter_states(fights, cache)

        # Canonicalize fighter names case-insensitively
        fighter_a = _canonical_fighter(raw_a, cache, fighter_states)
        fighter_b = _canonical_fighter(raw_b, cache, fighter_states)

        # Validate weight class (case-insensitive)
        if category not in WEIGHT_CLASSES:
            # try canonical mapping already done; if still not found warn
            print(f"warning: weight class '{args.weight_class}' not in known {WEIGHT_CLASSES} (using '{category}')", file=sys.stderr)

        # Check debut
        # For table we need current date
        current_date = datetime.now()

        # SHAP setup if requested
        shap_explainer = None
        shap_vals = None
        if args.explain and model_type in ("lightgbm", "stacking"):
            try:
                lgb_base = None
                if model_type == "stacking":
                    lgb_base = model.named_estimators_.get("lgbm")
                else:
                    lgb_base = model
                if lgb_base is not None:
                    shap_explainer = shap.TreeExplainer(lgb_base)
            except Exception as e:
                print(f"warning: SHAP not available: {e}", file=sys.stderr)

        def predict_order(f1, f2):
            X_enc = build_prediction_row(f1, f2, fighter_states, cache, current_date, category, priors, feature_meta)
            prob = model.predict_proba(X_enc)[0, 1]
            sv = None
            if shap_explainer is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray")
                    sv = shap_explainer.shap_values(X_enc)
                if isinstance(sv, list):
                    sv = sv[1][0]
                else:
                    sv = sv[0]
            return prob, sv

        prob_a_forward, shap_a = predict_order(fighter_a, fighter_b)
        prob_b_forward, shap_b = predict_order(fighter_b, fighter_a)
        prob_a = (prob_a_forward + (1.0 - prob_b_forward)) / 2.0
        prob_b = 1.0 - prob_a

        if args.json:
            out = {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "category": category,
                "rounds": max_rounds,
                "prob_a": round(float(prob_a), 6),
                "prob_b": round(float(prob_b), 6),
                "favorite": fighter_a if prob_a >= prob_b else fighter_b,
            }
            if args.explain and shap_a is not None:
                feat_names = feature_meta["feature_cols_final"]
                # choose favorite's shap
                shap_fav = shap_a if prob_a >= prob_b else shap_b
                pairs = sorted(zip(feat_names, shap_fav), key=lambda x: abs(x[1]), reverse=True)[:8]
                out["shap"] = [{"feature": f, "value": float(v)} for f, v in pairs]
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0

        # Table output
        print(f"\n  {'='*56}")
        print(f"  {fighter_a}  vs  {fighter_b}")
        print(f"  {'='*56}")
        print(f"  Weight class: {category}  Rounds: {max_rounds}")
        favorite, underdog = (fighter_a, fighter_b) if prob_a >= prob_b else (fighter_b, fighter_a)
        fav_prob, dog_prob = (prob_a, prob_b) if prob_a >= prob_b else (prob_b, prob_a)
        print(f"\n  PREDICTION")
        print(f"  {'-'*56}")
        print(f"\n  Favorite: {favorite} ({fav_prob*100:.1f}%)")
        print(f"  Underdog: {underdog} ({dog_prob*100:.1f}%)")

        # Comparative table (reuse predict.py logic simplified)

        def get_phys(name, key):
            v = cache.get(name, {}).get(key)
            return float(v) if v is not None else np.nan

        height_a, reach_a = get_phys(fighter_a, "height_cm"), get_phys(fighter_a, "reach_cm")
        height_b, reach_b = get_phys(fighter_b, "height_cm"), get_phys(fighter_b, "reach_cm")
        state_a = fighter_states.get(fighter_a, make_initial_state())
        state_b = fighter_states.get(fighter_b, make_initial_state())
        composite_params = feature_meta.get("composite_params")
        feat_a = compute_stats_from_state(state_a, fighter_a, cache, current_date, category=category, priors=priors, composite_params=composite_params)
        feat_b = compute_stats_from_state(state_b, fighter_b, cache, current_date, category=category, priors=priors, composite_params=composite_params)

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
        _line("Sig. Str. Absorbed/min", feat_a["sig_str_absorbed_per_min"], feat_b["sig_str_absorbed_per_min"])
        _line("Takedowns/15min", feat_a["td_avg_per_15min"], feat_b["td_avg_per_15min"])
        _line("TD Accuracy (%)", feat_a["td_accuracy"] * 100 if not np.isnan(feat_a["td_accuracy"]) else np.nan,
              feat_b["td_accuracy"] * 100 if not np.isnan(feat_b["td_accuracy"]) else np.nan, ".1f")
        print(sep)
        _line("Striking Strength", feat_a["striking"], feat_b["striking"])
        _line("Grappling Strength", feat_a["grappling"], feat_b["grappling"])
        _line("Durability", feat_a["durability"], feat_b["durability"])
        _line("Momentum", feat_a["momentum"], feat_b["momentum"])
        _line("Experience (opp quality)", feat_a["experience"], feat_b["experience"])
        print(eq_sep)
        print()

        if shap_explainer is not None and shap_a is not None:
            feat_names = feature_meta["feature_cols_final"]
            shap_fav = shap_a if prob_a >= prob_b else shap_b
            pairs = list(zip(feat_names, shap_fav))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"  SHAP - {favorite} (fav) vs {underdog} (dog)")
            print(f"  {'-'*56}")
            for feat, val in pairs[:8]:
                favor = favorite if val > 0 else underdog
                print(f"    {feat:<42s} {val:+7.4f}  -> {favor}")
            print()
        return 0

    # ---------- interactive mode ----------
    # Reuse logic from predict.py but with our loaded data

    def clear_screen():
        os.system("cls" if os.name == "nt" else "clear")

    def find_fighter(query, fighter_states, fighters_cache):
        q = query.lower().strip()
        names = set()
        for name in fighter_states:
            if q in name.lower():
                names.add(name)
        for name in fighters_cache:
            if q in name.lower():
                names.add(name)
        return sorted(names)

    def select_fighter(prompt, fighter_states, fighters_cache):
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

    print("=" * 60)
    print("  UFC FIGHT PREDICTOR")
    print("=" * 60)
    print(f"\n  {len(fights)} fights loaded")
    print(f"  {len(cache)} fighters in cache")
    priors = compute_priors(fights)
    print(f"  Model: {model_type}")
    print("  Building fighter histories...")
    fighter_states = build_fighter_states(fights, cache)
    print(f"  {len(fighter_states)} fighters with fight history")

    fighter_a = select_fighter("Enter first fighter name", fighter_states, cache)
    fighter_b = select_fighter("Enter second fighter name", fighter_states, cache)
    if fighter_a == fighter_b:
        print("\nBoth fighters are the same. Exiting.")
        return 0

    print("\n  Weight class:")
    for i, wc in enumerate(WEIGHT_CLASSES, 1):
        print(f"    {i}. {wc}")
    wc_input = input("\n  Select (number or name): ").strip()
    try:
        category = WEIGHT_CLASSES[int(wc_input) - 1]
    except (ValueError, IndexError):
        category = _canonical_weight_class(wc_input) or wc_input
    print(f"  Weight class: {category}")
    print("\n  Rounds:")
    print("    1. 3 rounds (non-title / prelim)")
    print("    2. 5 rounds (title / main event)")
    r_input = input("  Select (1/2, Enter=3): ").strip()
    max_rounds = 5 if r_input == "2" else 3
    clear_screen()

    print(f"\n  {'='*56}")
    print(f"  {fighter_a}  vs  {fighter_b}")
    print(f"  {'='*56}")
    print(f"  Weight class: {category}")

    current_date = datetime.now()

    # SHAP
    shap_explainer = None
    if model_type in ("lightgbm", "stacking"):
        lgb_base = None
        if model_type == "stacking":
            lgb_base = model.named_estimators_.get("lgbm")
        else:
            lgb_base = model
        if lgb_base is not None and hasattr(lgb_base, "named_steps"):
            lgb_base = lgb_base.named_steps.get("logreg", lgb_base)
        if lgb_base is not None:
            try:
                shap_explainer = shap.TreeExplainer(lgb_base)
            except Exception:
                shap_explainer = None

    def get_phys(name, key):
        v = cache.get(name, {}).get(key)
        return float(v) if v is not None else np.nan

    def predict_order(f1, f2):
        X_enc = build_prediction_row(f1, f2, fighter_states, cache, current_date, category, priors, feature_meta)
        prob = model.predict_proba(X_enc)[0, 1]
        shap_vals = None
        if shap_explainer is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray")
                sv = shap_explainer.shap_values(X_enc)
            if isinstance(sv, list):
                shap_vals = sv[1][0]
            else:
                shap_vals = sv[0]
        return prob, shap_vals

    prob_a_forward, shap_a_forward = predict_order(fighter_a, fighter_b)
    prob_b_forward, shap_b_forward = predict_order(fighter_b, fighter_a)
    prob_a = (prob_a_forward + (1.0 - prob_b_forward)) / 2.0
    prob_b = 1.0 - prob_a

    height_a, reach_a = get_phys(fighter_a, "height_cm"), get_phys(fighter_a, "reach_cm")
    height_b, reach_b = get_phys(fighter_b, "height_cm"), get_phys(fighter_b, "reach_cm")
    state_a = fighter_states.get(fighter_a, make_initial_state())
    state_b = fighter_states.get(fighter_b, make_initial_state())
    composite_params = feature_meta.get("composite_params")
    feat_a = compute_stats_from_state(state_a, fighter_a, cache, current_date, category=category, priors=priors, composite_params=composite_params)
    feat_b = compute_stats_from_state(state_b, fighter_b, cache, current_date, category=category, priors=priors, composite_params=composite_params)

    favorite, underdog = (fighter_a, fighter_b) if prob_a >= prob_b else (fighter_b, fighter_a)
    fav_prob, dog_prob = (prob_a, prob_b) if prob_a >= prob_b else (prob_b, prob_a)
    print(f"\n  PREDICTION")
    print(f"  {'-'*56}")
    print(f"\n  Favorite: {favorite} ({fav_prob*100:.1f}%)")
    print(f"  Underdog: {underdog} ({dog_prob*100:.1f}%)")

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
    streak_a = (f"{feat_a['current_win_streak']}W streak" if feat_a["current_win_streak"] > 0 else f"{feat_a['current_losing_streak']}L streak" if feat_a["current_losing_streak"] > 0 else "-")
    streak_b = (f"{feat_b['current_win_streak']}W streak" if feat_b["current_win_streak"] > 0 else f"{feat_b['current_losing_streak']}L streak" if feat_b["current_losing_streak"] > 0 else "-")
    _line("Streak", streak_a, streak_b)
    print(sep)
    _line("Striking Strength", feat_a["striking"], feat_b["striking"])
    _line("Grappling Strength", feat_a["grappling"], feat_b["grappling"])
    _line("Durability", feat_a["durability"], feat_b["durability"])
    _line("Momentum", feat_a["momentum"], feat_b["momentum"])
    _line("Experience (opp quality)", feat_a["experience"], feat_b["experience"])
    print(eq_sep)
    print()

    if shap_explainer is not None and shap_a_forward is not None:
        feat_names = feature_meta["feature_cols_final"]
        shap_fav = shap_a_forward if prob_a >= prob_b else shap_b_forward
        pairs = list(zip(feat_names, shap_fav))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"  SHAP - {favorite} (fav) vs {underdog} (dog)")
        print(f"  {'-'*56}")
        for feat, val in pairs[:8]:
            favor = favorite if val > 0 else underdog
            print(f"    {feat:<42s} {val:+7.4f}  -> {favor}")
        print()
    return 0


# ---------- event ----------

def handle_event(args):
    fights, cache, model_path, features_path = _load_data(args)
    event_name = args.event.strip()
    exact = args.exact
    as_json = args.json

    # Find event fights (case-insensitive for both modes)
    if exact:
        q = event_name.strip().lower()
        event_fights = [f for f in fights if f["event_name"].lower() == q]
    else:
        q = event_name.lower()
        event_fights = [f for f in fights if q in f["event_name"].lower()]

    if not event_fights:
        q = event_name.lower()
        all_events = sorted(set(f["event_name"] for f in fights))
        matches = [e for e in all_events if q in e.lower()]
        result = {"error": f"No fights found for event: {event_name}", "matched_events": matches[:20]}
        if as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"No fights found for event: {event_name}")
            if matches:
                print("Did you mean:")
                for m in matches[:10]:
                    print(f"  - {m}")
        return 1

    actual_event_name = event_fights[0]["event_name"]
    event_date = event_fights[0]["event_date"]

    model = joblib.load(model_path)
    feature_meta = joblib.load(features_path)


    event_dt = datetime.strptime(event_date, "%Y-%m-%d")
    fighter_states, priors = build_historical_context(fights, cache, event_dt)
    current_date = event_dt

    predictions = []
    skipped = []
    for fight in event_fights:
        f1 = fight["fighter_1"]
        f2 = fight["fighter_2"]
        category = fight["category"]
        winner = fight["winner"]
        if is_debut(f1, fighter_states, cache, event_date) or is_debut(f2, fighter_states, cache, event_date):
            skipped.append({"fighter_a": f1, "fighter_b": f2, "category": category, "winner": winner, "reason": "One or both fighters debuted at this event (0 UFC fights before it)"})
            continue
        try:
            prob_a, prob_b = predict_fight(f1, f2, category, fighter_states, cache, model, feature_meta, current_date, priors=priors)
            predictions.append({"fighter_a": f1, "fighter_b": f2, "prob_a": round(float(prob_a), 6), "prob_b": round(float(prob_b), 6), "category": category, "winner": winner})
        except Exception as e:
            predictions.append({"fighter_a": f1, "fighter_b": f2, "error": str(e), "winner": winner})

    output = {"event": actual_event_name, "date": event_date, "total_fights": len(predictions), "predictions": predictions, "skipped": skipped}

    if as_json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        _print_event_table(actual_event_name, event_date, predictions, skipped)
    return 0


# ---------- url ----------

def handle_url(args):
    url = args.url or args.url_flag
    if not url:
        print("error: provide an event URL as positional arg or --url", file=sys.stderr)
        return 2

    fights, cache, model_path, features_path = _load_data(args)
    as_json = args.json

    # Scrape event page
    if sync_playwright is None:
        print("error: playwright not installed. Run 'pip install playwright && playwright install chromium'", file=sys.stderr)
        return 1


    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    def parse_event_page(html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        h2 = soup.find("h2", class_="b-content__title")
        event_name = ""
        if h2:
            span = h2.find("span", class_="b-content__title-highlight")
            event_name = (span or h2).get_text(strip=True)
        event_date = None
        for li in soup.select("li.b-list__box-list-item"):
            text = li.get_text(" ", strip=True)
            if text.startswith("Date:"):
                date_str = text.replace("Date:", "").strip()
                try:
                    event_date = datetime.strptime(date_str, "%B %d, %Y")
                except ValueError:
                    event_date = None
        fights_list = []
        table = soup.find("table", class_="b-fight-details__table")
        if table:
            for row in table.select("tbody tr.b-fight-details__table-row"):
                cells = row.find_all("td")
                if len(cells) < 7:
                    continue
                fighter_cell = cells[1]
                links = fighter_cell.find_all("a")
                if len(links) < 2:
                    continue
                weight_cell = cells[6]
                category = weight_cell.get_text(strip=True)
                title_fight = bool(weight_cell.find("img") and "belt" in (weight_cell.find("img").get("src", "") or ""))
                fights_list.append({"fighter_1": links[0].get_text(strip=True), "fighter_2": links[1].get_text(strip=True), "category": category, "title_fight": title_fight})
        return {"event_name": event_name, "event_date": event_date, "fights": fights_list}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        try:
            page.goto(url, wait_until="commit", timeout=30000)
            page.wait_for_selector("table.b-fight-details__table", timeout=20000)
            page.wait_for_timeout(1500)
            html = page.content()
        finally:
            context.close()
            browser.close()

    event = parse_event_page(html)
    page_fights = event["fights"]
    event_date = event["event_date"]
    if not page_fights:
        print("No fights found on the page.", file=sys.stderr)
        return 1
    if event_date is None:
        print("Could not parse event date from the page.", file=sys.stderr)
        return 1

    model = joblib.load(model_path)
    feature_meta = joblib.load(features_path)

    fighter_states, priors = build_historical_context(fights, cache, event_date)

    results = []
    skipped = []
    for i, pf in enumerate(page_fights):
        f1, f2 = pf["fighter_1"], pf["fighter_2"]
        category = pf["category"]
        rounds = 5 if (i == 0 or pf["title_fight"]) else 3
        if f1 not in cache or f2 not in cache:
            skipped.append((f1, f2, category, "fighter not in cache"))
            continue
        state1 = fighter_states.get(f1, make_initial_state())
        state2 = fighter_states.get(f2, make_initial_state())
        tf1, tf2 = state1["total_fights"], state2["total_fights"]
        event_date_str = event_date.strftime("%Y-%m-%d")
        if is_debut(f1, fighter_states, cache, event_date_str) or is_debut(f2, fighter_states, cache, event_date_str):
            skipped.append((f1, f2, category, "0 UFC fights before the event"))
            continue
        prob_a, prob_b = predict_fight(f1, f2, category, fighter_states, cache, model, feature_meta, event_date, priors=priors)
        rec1 = (state1["wins"], state1["losses"], state1["draws"])
        rec2 = (state2["wins"], state2["losses"], state2["draws"])
        results.append((f1, f2, category, prob_a, prob_b, tf1, tf2, rounds, rec1, rec2))

    if as_json:
        out = {
            "event": event["event_name"] or url,
            "date": event_date.strftime("%Y-%m-%d"),
            "url": url,
            "predictions": [
                {"fighter_a": f1, "fighter_b": f2, "category": cat, "rounds": rnd, "prob_a": round(float(pa),6), "prob_b": round(float(pb),6),
                 "record_a": {"w": rec1[0], "l": rec1[1], "d": rec1[2], "total": tf1},
                 "record_b": {"w": rec2[0], "l": rec2[1], "d": rec2[2], "total": tf2}}
                for f1, f2, cat, pa, pb, tf1, tf2, rnd, rec1, rec2 in results
            ],
            "skipped": [{"fighter_a": f1, "fighter_b": f2, "category": cat, "reason": r} for f1,f2,cat,r in skipped],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        _print_url_table(event["event_name"], event_date, url, results, skipped)
    return 0
