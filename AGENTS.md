# AGENTS.md

## Project
UFC fight prediction pipeline: build events index → scrape fights → engineer features → train stacking ensemble → predict.

## Setup & Commands
- `.venv/Scripts/activate` to activate the venv (Windows).
- All scripts run from repo root so `data/...` and `models/...` resolve.
- No linting, typechecking, or test harness exists.
- Model artifacts use Git LFS (configured in `.gitattributes`).

## Pipeline Order
1. `python src/build_events_index.py` → `data/events_index.json`
2. `python src/scrape_ufc.py` → `data/fights.json`, `data/fighters_cache.json`
3. `python src/feature_engineering.py` → `data/dataset.csv`
4. `python src/train_model.py` → `models/ufc_stacking_ensemble.pkl` + `_meta.pkl`
5. `python src/predict.py` — interactive fighter-vs-fighter CLI
6. `python src/predict_event.py --event "UFC 328: ..."` — event-level JSON

## Architecture & Gotchas

### Model: Stacking Ensemble
The current model is **LightGBM + XGBoost + LogisticRegression with a meta-LogisticRegression**, wrapped in `ChronologicalStackingEnsemble` (`src/ensemble_utils.py`). That class uses `TimeSeriesSplit(n_splits=5)` for out-of-fold stacking features. The metadata pickle (`ufc_stacking_ensemble_meta.pkl`) stores `raw_feature_cols`, `cat_cols`, `feature_cols_final`, and `model_type="stacking"`.

### Scraping
Both `build_events_index.py` and `scrape_ufc.py` use **Playwright** (headless Chromium). You must have the browser installed (`playwright install chromium`). The scraper runs async, uses realistic UA headers, and includes random delays to avoid rate-limiting.

### Feature Engineering
- Fights are processed **chronologically** — each row uses only stats from fights before that date (no lookahead).
- Fighter A/B sides are **randomly assigned** per fight (seed=42 in `random.seed(42)`), not by position in the original JSON. This makes the dataset non-deterministic if the seed changes.
- Elo uses variable K-factor (96/64/40/24 by experience) and >1 year inactivity decay.

### Prediction
- `predict.py` and `predict_event.py` share the helper functions from their own modules (not from `feature_engineering.py`). Keep both in sync if you change computation logic.
- `predict_event.py` **filters out fights after the event date** before building fighter states — this is critical to prevent lookahead bias during event prediction.
- Both prediction scripts average predictions from both orderings (A→B and B→A) to remove order-dependent bias.
- SHAP explanations are active in `predict.py` (using `shap.TreeExplainer` on the LightGBM base estimator).

### Data Integrity
- `data/` and `models/` are generated artifacts. Don't edit by hand.
- Feature column changes in `train_model.py` must be mirrored in `predict.py` and `predict_event.py`.
- See `mejoras.md` for the (partially stale) roadmap — some items (SHAP, stacking ensemble) are already implemented.
