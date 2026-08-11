"""
Batch predict multiple UFC fights.

Usage:
  .venv/Scripts/python src/predict_batch.py "FighterA,FighterB,WeightClass" "FighterC,FighterD,WeightClass"
  .venv/Scripts/python src/predict_batch.py "FighterA,FighterB"

Each arg: Fighter1,Fighter2[,WeightClass]. WeightClass defaults to "Catch Weight".
"""
import sys
import json
import argparse
import joblib
from datetime import datetime

from config import (
    FIGHTS_PATH, FIGHTERS_CACHE_PATH, MODEL_PATH, FEATURE_COLS_PATH,
    CUTOFF_DATE, WEIGHT_CLASSES,
)
from fighter_engine import make_initial_state, build_fighter_states, predict_fight
from stats_utils import _prior_accum_init, _prior_accum_add, _get_current_priors


def main():
    parser = argparse.ArgumentParser(description="Batch predict UFC fights")
    parser.add_argument("fights", nargs="+", help="Fighter1,Fighter2[,WeightClass[,Rounds]]")
    args = parser.parse_args()

    parsed = []
    for arg in args.fights:
        parts = [p.strip() for p in arg.split(",")]
        if len(parts) == 2:
            f1, f2 = parts
            cat = "Catch Weight"
            rounds = 3
        elif len(parts) == 3:
            f1, f2, third = parts
            if third in ("3", "5"):
                cat = "Catch Weight"
                rounds = int(third)
            else:
                cat = third
                if cat not in WEIGHT_CLASSES:
                    print(f"Warning: '{cat}' is not a standard weight class, using as-is.")
                rounds = 3
        elif len(parts) == 4:
            f1, f2, cat, r_str = parts
            rounds = int(r_str) if r_str in ("3", "5") else 3
        else:
            print(f"Invalid format: '{arg}'. Use 'Fighter1,Fighter2[,WeightClass[,Rounds]]'.")
            sys.exit(1)
        parsed.append((f1, f2, cat, rounds))

    with open(FIGHTS_PATH) as f:
        fights = json.load(f)
    with open(FIGHTERS_CACHE_PATH) as f:
        fighters_cache = json.load(f)

    _prior_accum_init()
    for fight in fights:
        fight["_parsed_date"] = datetime.strptime(fight["event_date"], "%Y-%m-%d")
    for fight in sorted(fights, key=lambda f: f["_parsed_date"]):
        if fight["_parsed_date"] >= CUTOFF_DATE:
            _prior_accum_add(fight)
    priors = _get_current_priors()

    model = joblib.load(MODEL_PATH)
    feature_meta = joblib.load(FEATURE_COLS_PATH)

    fighter_states = build_fighter_states(fights, fighters_cache)
    current_date = datetime.now()

    results = []
    skipped = []
    warned = []

    for f1, f2, cat, rounds in parsed:
        state1 = fighter_states.get(f1, make_initial_state())
        state2 = fighter_states.get(f2, make_initial_state())
        tf1 = state1["total_fights"]
        tf2 = state2["total_fights"]

        if tf1 == 0 or tf2 == 0:
            skipped.append((f1, f2, tf1, tf2))
            continue

        if tf1 < 3 or tf2 < 3:
            warned.append((f1, f2, tf1, tf2))

        prob_a, prob_b = predict_fight(f1, f2, cat, fighter_states, fighters_cache,
                                       model, feature_meta, current_date, priors=priors)
        results.append((f1, f2, cat, prob_a, prob_b, tf1, tf2, rounds))

    # Output
    print()
    print("=" * 130)
    print("  BATCH PREDICTIONS")
    print("=" * 130)
    header = f"  {'#':<4s} {'Fighter A':<28s} {'Fighter B':<28s} {'Category':<22s} {'Prob A':<8s} {'Prob B':<8s} {'A fights':<9s} {'B fights':<9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    warned_set = {(f1, f2) for f1, f2, _, _ in warned}

    for i, (f1, f2, cat, prob_a, prob_b, tf1, tf2, rounds) in enumerate(results, 1):
        warn_flag = "  *" if (f1, f2) in warned_set else ""
        print(f"  {i:<4d} {f1:<28s} {f2:<28s} {cat:<22s} {prob_a*100:<7.1f}% {prob_b*100:<7.1f}% {tf1:<9d} {tf2:<9d}{warn_flag}")

    print("=" * 120)

    if warned:
        print("\n  * WARNING: One or both fighters have fewer than 3 UFC fights — prediction may be unreliable.")
        for f1, f2, tf1, tf2 in warned:
            print(f"    - {f1} ({tf1} fights) vs {f2} ({tf2} fights)")

    if skipped:
        print(f"\n  SKIPPED ({len(skipped)} fight(s) — no fight history for one or both fighters):")
        for f1, f2, tf1, tf2 in skipped:
            print(f"    - {f1} (0 fights) vs {f2} (0 fights)" if tf1 == 0 and tf2 == 0 else
                  f"    - {f1} ({tf1} fights) vs {f2} ({tf2} fights)")

    print()


if __name__ == "__main__":
    main()
