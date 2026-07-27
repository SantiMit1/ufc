# AGENTS.md

## Project
UFC fight prediction pipeline: build events index → scrape fights → engineer features → train stacking ensemble → predict.

## Setup & Commands
- `.venv/Scripts/activate` to activate the venv (Windows).
- All scripts run from repo root (`data/...`, `models/...` resolve relative to CWD).
- Scraper needs `playwright install chromium` before first run (`build_events_index.py` and `scrape_ufc.py` both use Playwright async headless Chromium).
- Model artifacts use Git LFS (`*.pkl filter=lfs` in `.gitattributes`).
- No linting, typechecking, or test harness exists.

## Pipeline Order (sequential)
1. `python src/build_events_index.py` → `data/events_index.json`
2. `python src/scrape_ufc.py` → `data/fights.json`, `data/fighters_cache.json`
3. `python src/feature_engineering.py` → `data/dataset.csv`
4. `python src/train_model.py` → `models/ufc_stacking_ensemble.pkl` + `ufc_stacking_ensemble_meta.pkl` + `models/ufc_method_model.pkl` + `models/ufc_round_model.pkl` + `models/stacking_ensemble_feature_importance.png` + `models/eval.txt`
5. `python src/predict.py` — interactive fighter-vs-fighter CLI (includes SHAP + method + round). Supports `--model`, `--features`, `--method-model`, `--round-model`. Prompts for rounds (3/5, default 3) and constrains predictions.
6. `python src/predict_event.py --event "UFC 328: ..."` — event-level JSON output with `method_probabilities` and `predicted_method`. No round prediction. Supports `--exact`, `--model-path`, `--features-path`, `--method-model-path`.

7. `python src/predict_batch.py "FighterA,FighterB,Category,5" "FighterC,FighterD"` — batch predictions table with Method/Round columns (with probabilities). Each arg is `Fighter1,Fighter2[,WeightClass[,Rounds]]` (defaults: WeightClass="Catch Weight", Rounds=3). Round column shows `-` when predicted method is DEC.
## Architecture & Gotchas

### Model: Stacking Ensemble
**LightGBM + XGBoost + LogisticRegression** with a meta-LogisticRegression, wrapped in `ChronologicalStackingEnsemble` (`src/ensemble_utils.py`). Uses `TimeSeriesSplit(n_splits=5)` for out-of-fold stacking. Metadata pickle stores `raw_feature_cols`, `cat_cols`, `numeric_cols`, `feature_cols_final`, and `model_type="stacking"`.

Three models trained sequentially:
- **Winner** (binary): `ufc_stacking_ensemble.pkl` — fighter A wins or not
- **Method** (multi-class, 3 classes): `ufc_method_model.pkl` — KO/SUB/DEC
- **Round** (multi-class, 5 classes): `ufc_round_model.pkl` — rounds 1-5

All three use the same feature pipeline (winner's `feature_cols_final`). The method/round models use `ChronologicalStackingEnsembleMultiClass` which flattens all class probabilities from base estimators as meta-features.

### Duplicated Computation Logic
`predict.py` and `predict_event.py` each maintain their own copies of helper functions (Elo, state tracking, feature computation). `predict_batch.py` imports them from `predict.py`. None import from `feature_engineering.py`. If you change feature computation logic, keep `predict.py` and `predict_event.py` in sync.

Script import quirks:
- `predict.py` and `predict_batch.py` use `sys.path.insert(0, str(Path(__file__).resolve().parent))` before `from ensemble_utils/stats_utils import ...`
- `predict_event.py` does **not** need `sys.path.insert` — it runs from `src/` so same-directory imports resolve automatically

### Feature Engineering
- Fights processed **chronologically** — each row uses only stats from fights before that date (no lookahead). Priors accumulated incrementally via `_prior_accum_add()`.
- Fighter A/B sides are **randomly assigned** per fight (`random.seed(42)` in `feature_engineering.py`). Dataset is non-deterministic if seed changes.
- Stats use Bayesian shrinkage toward population priors (`shrink_rate`, `shrink_proportion` in `stats_utils.py`). Priors per weight class; categories with <200 fights fall back to `"global"`.
- Elo uses variable K-factor (96/64/40/24 by experience) and >1 year inactivity decay.

### Prediction
- `predict_event.py` **filters out fights after the event date** before building fighter states — critical to prevent lookahead bias.
- All prediction scripts average predictions from both orderings (A→B and B→A) to remove order-dependent bias.
- SHAP explanations (`shap.TreeExplainer` on LightGBM base estimator) in `predict.py` only. `predict_event.py` outputs JSON and skips SHAP.

### Data Integrity
- `data/` and `models/` are generated artifacts. Don't edit by hand.
- Feature column changes in `train_model.py` must be mirrored in `predict.py` and `predict_event.py`.
- `dataset.csv` now includes `finish_type` (KO/SUB/DEC/OTHER) and `finish_round` (1-5) target columns.
