import random
import numpy as np
import pandas as pd

from config import DATASET_PATH, CUTOFF_DATE
from fighter_engine import (
    make_initial_state, compute_stats_from_state, classify_method, get_k_factor,
    apply_elo_decay, elo_update, update_state,
)
from stats_utils import PriorAccumulator, load_fights, load_fighter_cache, prepare_fights


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
    filtered_fights = prepare_fights(fights)
    discarded = total_raw - len(filtered_fights)
    print(f"Discarded {discarded} fights before {CUTOFF_DATE.date()}")

    # Initialize state dict
    fighter_state: dict[str, dict] = {}

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

            "striking_strength_diff": feat_a["striking"] - feat_b["striking"] if not (np.isnan(feat_a["striking"]) or np.isnan(feat_b["striking"])) else np.nan,
            "grappling_strength_diff": feat_a["grappling"] - feat_b["grappling"] if not (np.isnan(feat_a["grappling"]) or np.isnan(feat_b["grappling"])) else np.nan,
            "durability_diff": feat_a["durability"] - feat_b["durability"] if not (np.isnan(feat_a["durability"]) or np.isnan(feat_b["durability"])) else np.nan,
            "momentum_diff": feat_a["momentum"] - feat_b["momentum"] if not (np.isnan(feat_a["momentum"]) or np.isnan(feat_b["momentum"])) else np.nan,
            "experience_diff": feat_a["experience"] - feat_b["experience"] if not (np.isnan(feat_a["experience"]) or np.isnan(feat_b["experience"])) else np.nan,

            "winner": winner_label,
            "finish_type": finish_type_target if finish_type_target is not None else "OTHER",
            "finish_round": finish_round_val,
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
