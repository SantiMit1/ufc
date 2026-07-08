import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.10
MODEL_PATH = "models/ufc_model_lgbm.pkl"
FEATURE_COLS_PATH = "models/ufc_lgbm_feature_cols.pkl"

df = pd.read_csv("data/dataset.csv")
df = df[df["winner"].isin(["1", "2", 1, 2])].copy()
df["winner"] = df["winner"].astype(int)
df["event_date"] = pd.to_datetime(df["event_date"])
df.sort_values("event_date", inplace=True)
df.reset_index(drop=True, inplace=True)
y = (df["winner"] == 1).astype(int)

exclude_cols = {"fight_id", "event_date", "fighter_a_name", "fighter_b_name", "winner"}
diff_cols = [c for c in df.columns if c.endswith("_diff")]
other_feats = [
    "age_a", "age_b", "stance_a", "stance_b",
    "is_debut_a", "is_debut_b", "category",
    "recent_3_wins_a", "recent_3_wins_b",
    "recent_3_losses_a", "recent_3_losses_b",
    "recent_5_wins_a", "recent_5_wins_b",
    "recent_5_losses_a", "recent_5_losses_b",
    "recent_3_ko_loss_rate_a", "recent_3_ko_loss_rate_b",
    "recent_5_ko_loss_rate_a", "recent_5_ko_loss_rate_b",
    "decay_sig_per_min_a", "decay_sig_per_min_b",
    "decay_sig_absorbed_per_min_a", "decay_sig_absorbed_per_min_b",
    "decay_td_per_15min_a", "decay_td_per_15min_b",
    "avg_opp_elo_a", "avg_opp_elo_b",
    "avg_opp_elo_wins_a", "avg_opp_elo_wins_b",
]
raw_feature_cols = diff_cols + [c for c in other_feats if c not in diff_cols]
raw_feature_cols = [c for c in raw_feature_cols if c not in exclude_cols]

X_raw = df[raw_feature_cols].copy()
cat_cols = [c for c in ["category", "stance_a", "stance_b"] if c in X_raw.columns]
numeric_cols = [c for c in raw_feature_cols if c not in cat_cols]
for c in ["is_debut_a", "is_debut_b"]:
    if c in numeric_cols:
        X_raw[c] = X_raw[c].astype(int)

X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
feature_cols_final = list(X_encoded.columns)

n = len(df)
split_idx = int(n * (1 - TEST_SPLIT))
X_train_all = X_encoded.iloc[:split_idx].copy()
y_train_all = y.iloc[:split_idx]
X_test = X_encoded.iloc[split_idx:].copy()
y_test = y.iloc[split_idx:]

val_split = int(len(X_train_all) * (1 - VALIDATION_SPLIT))
X_train = X_train_all.iloc[:val_split].copy()
y_train = y_train_all.iloc[:val_split]
X_val = X_train_all.iloc[val_split:].copy()
y_val = y_train_all.iloc[val_split:]

for col in feature_cols_final:
    if col not in X_val.columns: X_val[col] = 0
    if col not in X_test.columns: X_test[col] = 0
X_val = X_val[feature_cols_final]
X_test = X_test[feature_cols_final]

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}  Features: {len(feature_cols_final)}")

# ─── HYPERPARAMETER SEARCH ────────────────────────────────────────────────────
param_grid = {
    "learning_rate": [0.01, 0.03, 0.05],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [10, 20, 50],
    "reg_alpha": [0.0, 0.01, 0.1],
    "reg_lambda": [0.0, 0.01, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

keys = list(param_grid.keys())
best_score = 0
best_params = {}
n_trials = 50
print(f"\nHyperparameter search ({n_trials} random trials)...")
print(f"{'trial':>5s} {'val_auc':>8s} {'val_ll':>8s} {'lr':>5s} {'leaves':>5s} {'min_child':>9s} {'reg_a':>6s} {'reg_l':>6s} {'sub':>5s} {'col':>5s}")
print(f"{'-'*75}")

for trial in range(n_trials):
    params = {k: np.random.choice(v) for k, v in param_grid.items()}
    m = lgb.LGBMClassifier(
        n_estimators=2000, verbose=-1, random_state=42, missing=float("nan"),
        class_weight="balanced", max_depth=-1,
        **params,
    )
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="logloss",
          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    yv_pred = m.predict_proba(X_val)[:, 1]
    va = roc_auc_score(y_val, yv_pred)
    vl = log_loss(y_val, yv_pred)
    if va > best_score:
        best_score = va
        best_params = params.copy()
        best_n = m.best_iteration_
    if trial < 20:
        print(f"{trial:5d} {va:8.4f} {vl:8.4f}  {params['learning_rate']:5.2f} {params['num_leaves']:5d} {params['min_child_samples']:9d} {params['reg_alpha']:6.2f} {params['reg_lambda']:6.2f} {params['subsample']:5.2f} {params['colsample_bytree']:5.2f}")

print(f"\nBest hyperparams (val_auc={best_score:.4f}, iters={best_n}):")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ─── CALIBRATION (Platt scaling) ──────────────────────────────────────────────

model = lgb.LGBMClassifier(
    n_estimators=2000, verbose=-1, random_state=42, missing=float("nan"),
    class_weight="balanced", max_depth=-1,
    **best_params,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="logloss",
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
best_n = model.best_iteration_

model.set_params(n_estimators=best_n)
model.fit(X_train, y_train)

# Uncalibrated vs calibrated comparison
y_prob_raw = model.predict_proba(X_test)[:, 1]
calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
calibrated.fit(X_val, y_val)
y_prob_cal = calibrated.predict_proba(X_test)[:, 1]

ll_raw = log_loss(y_test, y_prob_raw)
ll_cal = log_loss(y_test, y_prob_cal)
bs_raw = brier_score_loss(y_test, y_prob_raw)
bs_cal = brier_score_loss(y_test, y_prob_cal)
roc_raw = roc_auc_score(y_test, y_prob_raw)
roc_cal = roc_auc_score(y_test, y_prob_cal)

print(f"\n{'='*50}")
print("Calibration (Platt scaling)")
print(f"{'='*50}")
print(f"{'':20s} {'Raw':>8s} {'Calibrated':>12s}")
print(f"{'ROC-AUC':20s} {roc_raw:8.4f} {roc_cal:12.4f}")
print(f"{'Log Loss':20s} {ll_raw:8.4f} {ll_cal:12.4f}")
print(f"{'Brier score':20s} {bs_raw:8.4f} {bs_cal:12.4f}")

# ─── FINAL MODEL (on all training data) ────────────────────────────────────────
model_all = lgb.LGBMClassifier(
    n_estimators=best_n, verbose=-1, random_state=42, missing=float("nan"),
    class_weight="balanced", max_depth=-1,
    **best_params,
)
model_all.fit(X_train_all, y_train_all)

final_model = CalibratedClassifierCV(model_all, method='sigmoid', cv=5)
final_model.fit(X_train_all, y_train_all)

y_prob = final_model.predict_proba(X_test)[:, 1]
y_pred = final_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_prob)
bs = brier_score_loss(y_test, y_prob)
roc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*50}")
print("Test set evaluation (LightGBM + Platt scaling)")
print(f"{'='*50}")
print(f"Accuracy:     {acc:.4f}")
print(f"Log Loss:     {ll:.4f}")
print(f"Brier score:  {bs:.4f}")
print(f"ROC-AUC:      {roc:.4f}")
print(f"Best rounds:  {best_n}")

# Calibration (post-calibration)
bins = np.linspace(0.0, 1.0, 11)
bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, 9)
print(f"\n{'Post-Calibration':^50s}")
print(f"  {'Bin range':<12s} {'n':>5s} {'avg_prob':>9s} {'obs_freq':>9s}")
print(f"  {'-'*35}")
for i in range(10):
    mask = bin_indices == i
    nb = mask.sum()
    if nb == 0: continue
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f})  {nb:5d} {y_prob[mask].mean():9.3f} {y_test.values[mask].mean():9.3f}")

# ─── FEATURE IMPORTANCES ───────────────────────────────────────────────────────
importances = pd.DataFrame({
    "feature": feature_cols_final,
    "gain": model_all.booster_.feature_importance(importance_type="gain"),
    "split": model_all.booster_.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)
importances["pct"] = importances["gain"] / importances["gain"].sum() * 100

print(f"\n{'Feature Importances (by gain):':^60s}")
print(f"  {'Feature':<42s} {'Gain':>8s} {'%':>6s} {'Split':>6s}")
print(f"  {'-'*62}")
cum = 0
for _, r in importances.head(25).iterrows():
    cum += r["pct"]
    print(f"  {r['feature']:<42s} {r['gain']:>8.0f} {r['pct']:>5.1f}% {r['split']:>6.0f}")
print(f"  {'Top-25 cumulative':>42s} {cum:>14.1f}%")

# ─── PLOT ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    top_n = 20
    top = importances.head(top_n).copy()
    # Rename new columns for cleaner display
    rename_map = {
        "decay_sig_absorbed_per_min": "decay_sig_abs/min",
        "decay_sig_per_min": "decay_sig/min",
        "decay_td_per_15min": "decay_td/15min",
        "sig_str_landed_per_min": "sig_str_land/min",
        "sig_str_absorbed_per_min": "sig_str_abs/min",
        "sig_str_accuracy": "sig_acc",
        "td_avg_per_15min": "td/15min",
        "td_accuracy": "td_acc",
        "td_defense": "td_def",
        "sub_att_per_15min": "sub_att/15min",
        "ctrl_time_pct": "ctrl_time%",
        "days_since_last_fight": "days_since_fight",
        "avg_opp_elo_wins": "avg_opp_elo_W",
        "avg_opp_elo": "avg_opp_elo",
        "recent_3_ko_loss_rate": "r3_ko_loss_rate",
        "recent_5_ko_loss_rate": "r5_ko_loss_rate",
    }
    short = top["feature"].str.replace("_diff$", "_Δ", regex=True)
    short = short.str.replace(r"^(.*)_a$", r"\1_A", regex=True)
    short = short.str.replace(r"^(.*)_b$", r"\1_B", regex=True)
    for long, s in rename_map.items():
        short = short.str.replace(long, s, regex=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), top["gain"].values[::-1], color="#2b83ba", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(short.values[::-1])
    ax.invert_yaxis()
    ax.set_xlabel("Gain (total improvement from splits)")
    ax.set_title(f"LightGBM Feature Importance  |  Test ROC-AUC = {roc:.3f}")

    for i, v in enumerate(top["gain"].values[::-1]):
        ax.text(v + top["gain"].max() * 0.01, i, f"{v:.0f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig("models/lgbm_feature_importance.png", dpi=150)
    print(f"\nPlot saved to models/lgbm_feature_importance.png")
except ImportError:
    print("\nmatplotlib not installed, skipping plot")
except Exception as e:
    print(f"\nPlot failed: {e}")

# ─── SAVE ──────────────────────────────────────────────────────────────────────
joblib.dump(final_model, MODEL_PATH)
joblib.dump({
    "raw_feature_cols": raw_feature_cols,
    "cat_cols": cat_cols,
    "numeric_cols": numeric_cols,
    "feature_cols_final": feature_cols_final,
    "model_type": "lightgbm",
    "best_params": best_params,
    "test_roc_auc": roc,
    "test_accuracy": acc,
    "feature_importance": importances.to_dict(orient="records"),
}, FEATURE_COLS_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print(f"Metadata saved to {FEATURE_COLS_PATH}")
