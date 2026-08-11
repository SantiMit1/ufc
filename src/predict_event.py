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
import joblib
from datetime import datetime

from config import FIGHTS_PATH, FIGHTERS_CACHE_PATH, MODEL_PATH, FEATURE_COLS_PATH
from fighter_engine import build_fighter_states, predict_fight
from stats_utils import _prior_accum_init, _prior_accum_add, _get_current_priors


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
            prob_a, prob_b = predict_fight(f1, f2, category, fighter_states,
                                           fighters_cache, model, feature_meta,
                                           current_date, priors=priors)
            pred = {
                "fighter_a": f1,
                "fighter_b": f2,
                "prob_a": round(float(prob_a), 6),
                "prob_b": round(float(prob_b), 6),
                "category": category,
                "winner": winner,
            }
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
