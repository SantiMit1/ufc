import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
from ensemble_utils import ChronologicalStackingEnsemble, ChronologicalStackingEnsembleMultiClass
warnings.filterwarnings("ignore")

TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.10
MODEL_PATH = "models/ufc_stacking_ensemble.pkl"
FEATURE_COLS_PATH = "models/ufc_stacking_ensemble_meta.pkl"
df = pd.read_csv("data/dataset.csv")
df = df[df["winner"].isin(["1", "2", 1, 2])].copy()
df["winner"] = df["winner"].astype(int)
df["event_date"] = pd.to_datetime(df["event_date"])
df.sort_values("event_date", inplace=True)
df.reset_index(drop=True, inplace=True)
y = (df["winner"] == 1).astype(int)

exclude_cols = {"fight_id", "event_date", "fighter_a_name", "fighter_b_name", "winner"}

# Explicit feature selection: composites + standalone
selected_diff_cols = [
    "age_diff", "height_diff", "reach_diff", "elo_diff",
    "striking_strength_diff", "grappling_strength_diff",
    "durability_diff", "momentum_diff", "experience_diff",
]
other_feats = [
    "age_a", "age_b", "stance_a", "stance_b", "category",
]
raw_feature_cols = selected_diff_cols + [c for c in other_feats if c not in selected_diff_cols]
raw_feature_cols = [c for c in raw_feature_cols if c not in exclude_cols]

X_raw = df[raw_feature_cols].copy()
cat_cols = [c for c in ["category", "stance_a", "stance_b"] if c in X_raw.columns]
numeric_cols = [c for c in raw_feature_cols if c not in cat_cols]
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

# ─── LIGHTGBM HYPERPARAMETER SEARCH ───────────────────────────────────────────
lgb_param_grid = {
    "learning_rate": [0.01, 0.03, 0.05],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [10, 20, 50],
    "reg_alpha": [0.0, 0.01, 0.1],
    "reg_lambda": [0.0, 0.01, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

best_lgb_score = 0
best_lgb_params = {}
best_n_lgb = 0
n_lgb_trials = 50
print(f"\nLightGBM hyperparameter search ({n_lgb_trials} random trials)...")
print(f"{'trial':>5s} {'val_auc':>8s} {'val_ll':>8s} {'lr':>5s} {'leaves':>5s} {'min_child':>9s} {'reg_a':>6s} {'reg_l':>6s} {'sub':>5s} {'col':>5s}")
print(f"{'-'*75}")

for trial in range(n_lgb_trials):
    params = {k: np.random.choice(v) for k, v in lgb_param_grid.items()}
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
    if va > best_lgb_score:
        best_lgb_score = va
        best_lgb_params = params.copy()
        best_n_lgb = m.best_iteration_
    if trial < 20:
        print(f"{trial:5d} {va:8.4f} {vl:8.4f}  {params['learning_rate']:5.2f} {params['num_leaves']:5d} {params['min_child_samples']:9d} {params['reg_alpha']:6.2f} {params['reg_lambda']:6.2f} {params['subsample']:5.2f} {params['colsample_bytree']:5.2f}")

print(f"\nBest LightGBM hyperparams (val_auc={best_lgb_score:.4f}, iters={best_n_lgb}):")
for k, v in best_lgb_params.items():
    print(f"  {k}: {v}")


# ─── XGBoost HYPERPARAMETER SEARCH ────────────────────────────────────────────
xgb_param_grid = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 6],
    "min_child_weight": [1, 2, 5],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0.0, 0.01, 0.1],
    "reg_lambda": [0.0, 0.5, 1.0],
}

best_xgb_score = 0
best_xgb_params = {}
best_n_xgb = 0
n_xgb_trials = 20
print(f"\nXGBoost hyperparameter search ({n_xgb_trials} random trials)...")
print(f"{'trial':>5s} {'val_auc':>8s} {'val_ll':>8s} {'lr':>5s} {'max_d':>5s} {'min_child':>9s} {'sub':>5s} {'col':>5s} {'reg_a':>6s} {'reg_l':>6s}")
print(f"{'-'*80}")

for trial in range(n_xgb_trials):
    params = {k: np.random.choice(v) for k, v in xgb_param_grid.items()}
    m = xgb.XGBClassifier(
        n_estimators=1000,
        early_stopping_rounds=30,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        missing=np.nan,
        **params,
    )
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    yv_pred = m.predict_proba(X_val)[:, 1]
    va = roc_auc_score(y_val, yv_pred)
    vl = log_loss(y_val, yv_pred)
    n_est = m.best_iteration
    if va > best_xgb_score:
        best_xgb_score = va
        best_xgb_params = params.copy()
        best_n_xgb = n_est
    if trial < 10:
        print(f"{trial:5d} {va:8.4f} {vl:8.4f}  {params['learning_rate']:5.2f} {params['max_depth']:5d} {params['min_child_weight']:9d} {params['subsample']:5.2f} {params['colsample_bytree']:5.2f} {params['reg_alpha']:6.2f} {params['reg_lambda']:6.2f}")

print(f"\nBest XGBoost hyperparams (val_auc={best_xgb_score:.4f}, iters={best_n_xgb}):")
for k, v in best_xgb_params.items():
    print(f"  {k}: {v}")


# ─── BUILD BASE MODELS FOR ENSEMBLE ───────────────────────────────────────────
def build_lgbm_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=best_n_lgb,
        verbose=-1,
        random_state=42,
        missing=float("nan"),
        class_weight="balanced",
        max_depth=-1,
        **best_lgb_params,
    )


def build_xgb_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=max(best_n_xgb, 50),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        missing=np.nan,
        **best_xgb_params,
    )


lr_base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")),
])

stacking_model = ChronologicalStackingEnsemble(
    estimators=[
        ("lgbm", build_lgbm_model()),
        ("xgb", build_xgb_model()),
        ("logreg", lr_base),
    ],
    final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
    cv=TimeSeriesSplit(n_splits=5),
)

print(f"\n{'='*50}")
print("Training stacked ensemble (LightGBM + XGBoost + Logistic Regression)")
print(f"{'='*50}")
stacking_model.fit(X_train_all, y_train_all)

final_model = stacking_model
y_prob = final_model.predict_proba(X_test)[:, 1]
y_pred = final_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_prob)
bs = brier_score_loss(y_test, y_prob)
roc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*50}")
print("Test set evaluation (stacked ensemble)")
print(f"{'='*50}")
print(f"Accuracy:     {acc:.4f}")
print(f"Log Loss:     {ll:.4f}")
print(f"Brier score:  {bs:.4f}")
print(f"ROC-AUC:      {roc:.4f}")
print(f"LGB rounds:   {best_n_lgb}  |  XGB rounds:  {best_n_xgb}")

# Probability calibration of the ensemble predictions
bins = np.linspace(0.0, 1.0, 11)
bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, 9)
print(f"\n{'Ensemble Calibration':^50s}")
print(f"  {'Bin range':<12s} {'n':>5s} {'avg_prob':>9s} {'obs_freq':>9s}")
print(f"  {'-'*35}")
for i in range(10):
    mask = bin_indices == i
    nb = mask.sum()
    if nb == 0: continue
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f})  {nb:5d} {y_prob[mask].mean():9.3f} {y_test.values[mask].mean():9.3f}")

# ─── FEATURE IMPORTANCES ───────────────────────────────────────────────────────
model_all = final_model.named_estimators_["lgbm"]
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
    ax.set_title(f"Stacked Ensemble Feature Importance  |  Test ROC-AUC = {roc:.3f}")

    for i, v in enumerate(top["gain"].values[::-1]):
        ax.text(v + top["gain"].max() * 0.01, i, f"{v:.0f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig("models/stacking_ensemble_feature_importance.png", dpi=150)
    print(f"\nPlot saved to models/stacking_ensemble_feature_importance.png")
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
    "model_type": "stacking",
    "best_lgb_params": best_lgb_params,
    "best_n_lgb": best_n_lgb,
    "best_xgb_params": best_xgb_params,
    "best_n_xgb": best_n_xgb,
    "ensemble_estimators": ["lightgbm", "xgboost", "logistic_regression"],
    "ensemble_cv": "TimeSeriesSplit(n_splits=5)",
    "test_roc_auc": roc,
    "test_accuracy": acc,
    "feature_importance": importances.to_dict(orient="records"),
}, FEATURE_COLS_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print(f"Metadata saved to {FEATURE_COLS_PATH}")

# ═══════════════════════════════════════════════════════════════════════════════
# METHOD MODEL (Multi-class: KO, SUB, DEC)
# ═══════════════════════════════════════════════════════════════════════════════
METHOD_MODEL_PATH = "models/ufc_method_model.pkl"
ROUND_MODEL_PATH = "models/ufc_round_model.pkl"

def _build_multiclass_lgbm(n_classes):
    return lgb.LGBMClassifier(
        n_estimators=best_n_lgb,
        verbose=-1, random_state=42, missing=float("nan"),
        class_weight="balanced", max_depth=-1,
        objective="multiclass", num_class=n_classes,
        metric="multi_logloss",
        **best_lgb_params,
    )

def _build_multiclass_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=max(best_n_xgb, 50),
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist", random_state=42, n_jobs=-1,
        missing=np.nan,
        **best_xgb_params,
    )

def _build_multiclass_lr(n_classes):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="lbfgs",
        )),
    ])

# ─── METHOD: finish type → KO=0, SUB=1, DEC=2 ─────────────────────────────
print(f"\n{'='*50}")
print("Training method model (KO / SUB / DEC)")
print(f"{'='*50}")

method_valid = df["finish_type"].isin(["KO", "SUB", "DEC"])
X_method = X_encoded[method_valid.values].copy()
y_method = df.loc[method_valid, "finish_type"].map({"KO": 0, "SUB": 1, "DEC": 2}).values

n_method = len(y_method)
split_idx_method = int(n_method * (1 - TEST_SPLIT))
X_method_train = X_method.iloc[:split_idx_method]
y_method_train = y_method[:split_idx_method]
X_method_test = X_method.iloc[split_idx_method:]
y_method_test = y_method[split_idx_method:]

print(f"Method samples: {n_method}  (train={len(y_method_train)}, test={len(y_method_test)})")
print(f"  KO={int((y_method==0).sum())}  SUB={int((y_method==1).sum())}  DEC={int((y_method==2).sum())}")

method_ensemble = ChronologicalStackingEnsembleMultiClass(
    estimators=[
        ("lgbm", _build_multiclass_lgbm(3)),
        ("xgb", _build_multiclass_xgb(3)),
        ("logreg", _build_multiclass_lr(3)),
    ],
    final_estimator=LogisticRegression(
        max_iter=3000, class_weight="balanced", solver="lbfgs",
    ),
    cv=TimeSeriesSplit(n_splits=5),
    n_classes=3,
)
method_ensemble.fit(X_method_train, y_method_train)

y_method_pred = method_ensemble.predict(X_method_test)
method_acc = accuracy_score(y_method_test, y_method_pred)
print(f"Method test accuracy: {method_acc:.4f}")

joblib.dump(method_ensemble, METHOD_MODEL_PATH)
print(f"Method model saved to {METHOD_MODEL_PATH}")

# ─── ROUND MODEL (Multi-class: rounds 1-5) ─────────────────────────────────
print(f"\n{'='*50}")
print("Training round model (Rounds 1-5)")
print(f"{'='*50}")

round_valid = df["finish_round"].between(1, 5)
X_round = X_encoded[round_valid.values].copy()
y_round = (df.loc[round_valid, "finish_round"].astype(int) - 1).values

n_round = len(y_round)
split_idx_round = int(n_round * (1 - TEST_SPLIT))
X_round_train = X_round.iloc[:split_idx_round]
y_round_train = y_round[:split_idx_round]
X_round_test = X_round.iloc[split_idx_round:]
y_round_test = y_round[split_idx_round:]

print(f"Round samples: {n_round}  (train={len(y_round_train)}, test={len(y_round_test)})")
for r in range(5):
    print(f"  Round {r+1}={int((y_round==r).sum())}")

round_ensemble = ChronologicalStackingEnsembleMultiClass(
    estimators=[
        ("lgbm", _build_multiclass_lgbm(5)),
        ("xgb", _build_multiclass_xgb(5)),
        ("logreg", _build_multiclass_lr(5)),
    ],
    final_estimator=LogisticRegression(
        max_iter=3000, class_weight="balanced", solver="lbfgs",
    ),
    cv=TimeSeriesSplit(n_splits=5),
    n_classes=5,
)
round_ensemble.fit(X_round_train, y_round_train)

y_round_pred = round_ensemble.predict(X_round_test)
round_acc = accuracy_score(y_round_test, y_round_pred)
print(f"Round test accuracy: {round_acc:.4f}")

joblib.dump(round_ensemble, ROUND_MODEL_PATH)
print(f"Round model saved to {ROUND_MODEL_PATH}")

print(f"\n{'='*50}")
print("All models trained successfully!")
print(f"{'='*50}")
