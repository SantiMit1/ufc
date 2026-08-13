import random
import numpy as np
import pandas as pd

from config import DATASET_PATH, CUTOFF_DATE
from fighter_engine import FightStateEngine, compute_stats_from_state, classify_method, compute_feature_diffs
from stats_utils import PriorAccumulator, load_fights, load_fighter_cache


SEED = 42


def main():
    random.seed(SEED)

    print("Loading fights...")
    fights = load_fights()

    print("Loading fighters cache...")
    fighters_cache = load_fighter_cache()

    total_raw = len(fights)

    # Initialize prior accumulator for incremental (no-lookahead) priors
    prior_accum = PriorAccumulator()

    # Parse dates, filter by cutoff and sort chronologically
    engine = FightStateEngine(fights)
    discarded = total_raw - len(engine.filtered)
    print(f"Discarded {discarded} fights before {CUTOFF_DATE.date()}")

    rows = []
    debut_a_count = 0
    debut_b_count = 0
    unknown_stance_count = 0

    for idx, fight in enumerate(engine, 1):
        f1_name = fight["fighter_1"]
        f2_name = fight["fighter_2"]

        fighter_state = engine.state

        # Randomly assign fighter_a/fighter_b
        if random.random() < 0.5:
            a_name, b_name = f1_name, f2_name
            a_is_f1 = True
            b_is_f1 = False
        else:
            a_name, b_name = f2_name, f1_name
            a_is_f1 = False
            b_is_f1 = True

        # Compute pre-fight features using priors from fights BEFORE this one
        priors = prior_accum.priors()
        feat_a = compute_stats_from_state(fighter_state[a_name], a_name, fighters_cache, fight["_parsed_date"], category=fight["category"], priors=priors, fight=fight, is_fighter_1=a_is_f1)
        feat_b = compute_stats_from_state(fighter_state[b_name], b_name, fighters_cache, fight["_parsed_date"], category=fight["category"], priors=priors, fight=fight, is_fighter_1=b_is_f1)

        # Add fight stats to prior accumulator AFTER computing features (no lookahead)
        prior_accum.add(fight)

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

        # Determine finish type and round for target labels
        _, _, finish_type_target = classify_method(
            fight["method"], fight["winner"], fight["fighter_1"], fight["fighter_2"]
        )
        fight_round_raw = fight.get("round")
        finish_round_val = int(fight_round_raw) if fight_round_raw is not None else 0

        row = {
            "fight_id": f"fight_{idx:05d}",
            "event_date": fight["event_date"],
            "category": fight.get("category", ""),
            "fighter_a_name": a_name,
            "fighter_b_name": b_name,
            "age_a": feat_a["age"],
            "age_b": feat_b["age"],
            "age_diff": feat_a["age"] - feat_b["age"] if not (np.isnan(feat_a["age"]) or np.isnan(feat_b["age"])) else np.nan,
            "stance_a": feat_a["stance"],
            "stance_b": feat_b["stance"],
            "height_diff": height_a - height_b if not (np.isnan(height_a) or np.isnan(height_b)) else np.nan,
            "reach_diff": reach_a - reach_b if not (np.isnan(reach_a) or np.isnan(reach_b)) else np.nan,
        }
        row.update(compute_feature_diffs(feat_a, feat_b))
        row.update({
            "winner": winner_label,
            "finish_type": finish_type_target if finish_type_target is not None else "OTHER",
            "finish_round": finish_round_val,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(DATASET_PATH, index=False)

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
    print(f"\nDataset saved to {DATASET_PATH}")


if __name__ == "__main__":
    main()
