import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
import joblib

# ─── CONFIG ─────────────────────────────────────────────────────────────────
TEST_SPLIT = 0.15
MODEL_PATH = 'models/ufc_model_logreg_full.pkl'
SCALER_PATH = 'models/ufc_scaler_logreg_full.pkl'
FEATURE_COLS_PATH = 'models/ufc_logreg_full_feature_cols.pkl'

# ─── LOAD ────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/dataset.csv')

df = df[df['winner'].isin(['1', '2', 1, 2])].copy()
df['winner'] = df['winner'].astype(int)
df['event_date'] = pd.to_datetime(df['event_date'])
df.sort_values('event_date', inplace=True)
df.reset_index(drop=True, inplace=True)

y = (df['winner'] == 1).astype(int)

# ─── FEATURE SELECTION (full set, same columns as the LightGBM scripts) ────
exclude_cols = {'fight_id', 'event_date', 'fighter_a_name', 'fighter_b_name', 'winner'}
diff_cols = [c for c in df.columns if c.endswith('_diff')]
other_feats = ['age_a', 'age_b', 'stance_a', 'stance_b',
               'is_debut_a', 'is_debut_b', 'category']
if 'title_bout' in df.columns:
    other_feats.append('title_bout')

raw_feature_cols = diff_cols + [c for c in other_feats if c not in diff_cols]
raw_feature_cols = [c for c in raw_feature_cols if c not in exclude_cols]

print(f'Raw feature columns ({len(raw_feature_cols)}): {raw_feature_cols}')

X_raw = df[raw_feature_cols].copy()

# ─── HANDLE CATEGORICALS: one-hot encode ────────────────────────────────────
cat_cols = [c for c in ['category', 'stance_a', 'stance_b'] if c in X_raw.columns]
numeric_cols = [c for c in raw_feature_cols if c not in cat_cols]

# is_debut_a/b are booleans -> treat as numeric 0/1
for c in ['is_debut_a', 'is_debut_b']:
    if c in numeric_cols:
        X_raw[c] = X_raw[c].astype(int)

if 'title_bout' in numeric_cols:
    X_raw['title_bout'] = X_raw['title_bout'].astype(int)

# Impute NaN in numeric columns with the column median.
# NOTE: unlike LightGBM, plain logistic regression cannot handle NaN
# natively, so missing values (mostly from debut fighters / 0-attempt
# ratios) are filled with the median. This loses some of the "missingness
# is informative" signal that LightGBM could use directly; if that
# matters, consider adding explicit *_is_missing indicator columns.
medians = {}
for c in numeric_cols:
    med = X_raw[c].median()
    medians[c] = med
    X_raw[c] = X_raw[c].fillna(med)

X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
feature_cols_final = list(X_encoded.columns)

print(f'Feature columns after one-hot ({len(feature_cols_final)})')

# ─── TEMPORAL SPLIT ─────────────────────────────────────────────────────────
n = len(df)
split_idx = int(n * (1 - TEST_SPLIT))

train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

X_train = X_encoded.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test  = X_encoded.iloc[split_idx:]
y_test  = y.iloc[split_idx:]

print(f'\nTrain size: {len(X_train)}  ({train_df["event_date"].min().date()} -> {train_df["event_date"].max().date()})')
print(f'Test size:  {len(X_test)}   ({test_df["event_date"].min().date()} -> {test_df["event_date"].max().date()})')
print(f'Train class 1 ratio: {y_train.mean():.3f}')
print(f'Test  class 1 ratio: {y_test.mean():.3f}')

# ─── SCALE (fit on train only) ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─── MODEL ──────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=2000, C=1.0)
model.fit(X_train_scaled, y_train)

y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
ll  = log_loss(y_test, y_prob)
bs  = brier_score_loss(y_test, y_prob)
roc = roc_auc_score(y_test, y_prob)

print(f'\n{"-"*50}')
print('Test set evaluation (Logistic Regression, full feature set)')
print(f'{"-"*50}')
print(f'Accuracy:    {acc:.4f}')
print(f'Log Loss:    {ll:.4f}')
print(f'Brier score: {bs:.4f}')
print(f'ROC-AUC:     {roc:.4f}')

# Calibration check
bins = np.linspace(0.0, 1.0, 11)
bin_indices = np.digitize(y_prob, bins) - 1
bin_indices = np.clip(bin_indices, 0, 9)

print(f'\nCalibration (prob bins vs observed freq):')
print(f'  Bin range     |  n   | avg_prob | obs_freq')
print(f'  {"-"*42}')
for i in range(10):
    mask = bin_indices == i
    n_bin = mask.sum()
    if n_bin == 0:
        continue
    avg_prob = y_prob[mask].mean()
    obs_freq = y_test.values[mask].mean()
    print(f'  [{bins[i]:.1f}-{bins[i+1]:.1f})   | {n_bin:4d} |   {avg_prob:.3f}  |  {obs_freq:.3f}')

# ─── COEFFICIENTS ────────────────────────────────────────────────────────────
coef_df = pd.DataFrame({
    'feature': feature_cols_final,
    'coef': model.coef_[0],
}).sort_values('coef', key=abs, ascending=False)

print(f'\nCoefficients (standardized), sorted by |coef|:')
print(coef_df.to_string(index=False))

# ─── SAVE MODEL + SCALER + METADATA ─────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump({
    'raw_feature_cols': raw_feature_cols,
    'cat_cols': cat_cols,
    'numeric_cols': numeric_cols,
    'medians': medians,
    'feature_cols_final': feature_cols_final,
}, FEATURE_COLS_PATH)

print(f'\nModel saved to {MODEL_PATH}')
print(f'Scaler saved to {SCALER_PATH}')
print(f'Feature metadata saved to {FEATURE_COLS_PATH}')