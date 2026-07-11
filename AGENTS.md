# AGENTS.md

## Project
UFC fight prediction pipeline: build events index → scrape fights → engineer features → train stacking ensemble → predict.

## Setup & Commands
- `.venv/Scripts/activate` to activate the venv (Windows).
- All scripts run from repo root so `data/...` and `models/...` resolve (script's `BASE_DIR = Path(__file__).parent.parent`).
- Scraper needs `playwright install chromium` before first run (both `build_events_index.py` and `scrape_ufc.py` use Playwright async headless Chromium).
- Model artifacts use Git LFS (`*.pkl filter=lfs` in `.gitattributes`).
- No linting, typechecking, or test harness exists.

## Pipeline Order
1. `python src/build_events_index.py` → `data/events_index.json`
2. `python src/scrape_ufc.py` → `data/fights.json`, `data/fighters_cache.json`
3. `python src/feature_engineering.py` → `data/dataset.csv`
4. `python src/train_model.py` → `models/ufc_stacking_ensemble.pkl` + `ufc_stacking_ensemble_meta.pkl`
5. `python src/predict.py` — interactive fighter-vs-fighter CLI (includes SHAP)
6. `python src/predict_event.py --event "UFC 328: ..."` — event-level JSON (no SHAP)
7. `python bets.py` — value-bet calculator using `predict_event_valuebets()` from `predict.py`
8. `python evaluate_valuebets.py` — evaluate model performance on past events (edit `EVALUATION_FIGHTS` list in the script). Same output format as `bets.py` plus MODEL EVALUATION summary. No lookahead: filters fights to before each `event_date` before building states.

## Architecture & Gotchas

### Model: Stacking Ensemble
**LightGBM + XGBoost + LogisticRegression** with a meta-LogisticRegression, wrapped in `ChronologicalStackingEnsemble` (`src/ensemble_utils.py`). Uses `TimeSeriesSplit(n_splits=5)` for out-of-fold stacking features. The metadata pickle stores `raw_feature_cols`, `cat_cols`, `numeric_cols`, `feature_cols_final`, and `model_type="stacking"`.

### Duplicated Computation Logic
`predict.py` and `predict_event.py` each maintain their own copies of helper functions (Elo, state tracking, feature computation). They do **not** import from `feature_engineering.py`. If you change feature computation logic, keep all three in sync. Prediction scripts use `sys.path.insert(0, ...)` at the top to import local `ensemble_utils` and `stats_utils`.

### Feature Engineering
- Fights processed **chronologically** — each row uses only stats from fights before that date (no lookahead). Priors are accumulated incrementally via `_prior_accum_add()`.
- Fighter A/B sides are **randomly assigned** per fight (`random.seed(42)`), not by position in the original JSON. Dataset is non-deterministic if seed changes.
- Stats use Bayesian shrinkage toward population priors (`shrink_rate`, `shrink_proportion` in `stats_utils.py`). Priors are computed per weight class; categories with <200 fights fall back to `"global"`.
- Elo uses variable K-factor (96/64/40/24 by experience) and >1 year inactivity decay.

### Prediction
- `predict_event.py` **filters out fights after the event date** before building fighter states — critical to prevent lookahead bias.
- Both prediction scripts average predictions from both orderings (A→B and B→A) to remove order-dependent bias.
- SHAP explanations (`shap.TreeExplainer` on LightGBM base estimator) are active in `predict.py` only. `predict_event.py` outputs JSON and skips SHAP.

### Data Integrity
- `data/` and `models/` are generated artifacts. Don't edit by hand.
- Feature column changes in `train_model.py` must be mirrored in `predict.py` and `predict_event.py`.
