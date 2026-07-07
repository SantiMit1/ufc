import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
import joblib

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
    if col not in X_val.columns:
        X_val[col] = 0
    if col not in X_test.columns:
        X_test[col] = 0
X_val = X_val[feature_cols_final]
X_test = X_test[feature_cols_final]

print(f"Train size: {len(X_train)}")
print(f"Val size:   {len(X_val)}")
print(f"Test size:  {len(X_test)}")
print(f"Features:   {len(feature_cols_final)}")
print(f"Train class 1 ratio: {y_train.mean():.3f}")

model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.01,
    reg_lambda=0.01,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
    missing=float("nan"),
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="logloss",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_prob)
bs = brier_score_loss(y_test, y_prob)
roc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*50}")
print("Test set evaluation (LightGBM)")
print(f"{'='*50}")
print(f"Accuracy:    {acc:.4f}")
print(f"Log Loss:    {ll:.4f}")
print(f"Brier score: {bs:.4f}")
print(f"ROC-AUC:     {roc:.4f}")

bins = np.linspace(0.0, 1.0, 11)
bin_indices = np.digitize(y_prob, bins) - 1
bin_indices = np.clip(bin_indices, 0, 9)

print(f"\nCalibration (prob bins vs observed freq):")
print(f"  Bin range     |  n   | avg_prob | obs_freq")
print(f"  {'-'*42}")
for i in range(10):
    mask = bin_indices == i
    n_bin = mask.sum()
    if n_bin == 0:
        continue
    avg_prob = y_prob[mask].mean()
    obs_freq = y_test.values[mask].mean()
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f})   | {n_bin:4d} |   {avg_prob:.3f}  |  {obs_freq:.3f}")

importances = pd.DataFrame({
    "feature": feature_cols_final,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print(f"\nFeature importances (top 20):")
for _, row in importances.head(20).iterrows():
    print(f"  {row['feature']:<45s} {row['importance']:>6.0f}")

n_estimators_used = model.best_iteration_ or model.n_estimators
print(f"\nBest iteration: {n_estimators_used}")

model.set_params(n_estimators=n_estimators_used)
model.fit(X_train_all, y_train_all)

joblib.dump(model, MODEL_PATH)
joblib.dump({
    "raw_feature_cols": raw_feature_cols,
    "cat_cols": cat_cols,
    "numeric_cols": numeric_cols,
    "feature_cols_final": feature_cols_final,
    "model_type": "lightgbm",
}, FEATURE_COLS_PATH)

print(f"\nModel saved to {MODEL_PATH}")
