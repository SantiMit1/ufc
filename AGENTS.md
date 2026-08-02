# AGENTS.md

## Pipeline (sequential)
1. `python src/build_events_index.py` → `data/events_index.json` (skips upcoming/unfinished events)
2. `python src/scrape_ufc.py` → `data/fights.json`, `data/fighters_cache.json`
3. `python src/feature_engineering.py` → `data/dataset.csv`
4. `python src/train_model.py` → `models/ufc_stacking_ensemble.pkl` + `_meta.pkl` + feature importance PNG (run manually)
5. `python src/predict.py` — interactive CLI (SHAP). Args: `--model`, `--features`.
6. `python src/predict_event.py --event "UFC 328: ..."` — event JSON with winner probabilities. Args: `--exact`, `--model-path`, `--features-path`.
7. `python src/predict_batch.py "FighterA,FighterB,Category,5" "FighterC,FighterD"` — batch table. Each arg: `F1,F2[,WeightClass[,Rounds]]`. Rounds defaults 3, WeightClass defaults "Catch Weight". Import helpers from `predict.py`. No model-path CLI flags (hardcoded paths).

To run the pipeline (skip step 4, run it by hand):
```bash
.venv/Scripts/activate
python src/build_events_index.py
python src/scrape_ufc.py
python src/feature_engineering.py
```

## Setup
- `.venv/Scripts/activate` (Windows). All scripts from repo root.
- Scraper needs `playwright install chromium` before first run (both `build_events_index.py` and `scrape_ufc.py` use Playwright async headless Chromium).
- Model artifacts use Git LFS (`*.pkl filter=lfs` in `.gitattributes`). No linting, typechecking, or test harness.

## Architecture

### Model: Stacking Ensemble
`ChronologicalStackingEnsemble` (`src/ensemble_utils.py`) — **LightGBM + XGBoost + LogisticRegression** (Pipeline: imputer→scaler→LR) with a meta-LR, `TimeSeriesSplit(n_splits=5)` OOF. Metadata stores `raw_feature_cols`, `cat_cols`, `numeric_cols`, `feature_cols_final`, `model_type="stacking"`.

Single model, trained on the shared feature pipeline:
- **Winner** (binary): `ufc_stacking_ensemble.pkl` — prob fighter A wins

### Duplicated Computation
`predict.py` and `predict_event.py` each maintain **their own copies** of Elo, state tracking, and feature computation. `predict_batch.py` imports from `predict.py`. None import from `feature_engineering.py`. If you change feature logic, sync `predict.py` and `predict_event.py` manually.

Import quirks:
- `predict.py` and `predict_batch.py` use `sys.path.insert(0, ...)` before `from ensemble_utils/stats_utils import ...`
- `predict_event.py` does **not** — it runs from `src/` so same-directory imports resolve automatically

### Feature Engineering
- Fights processed **chronologically** with no lookahead. Priors accumulate incrementally via `_prior_accum_add()`.
- Fighter A/B sides **randomly assigned** per fight (`random.seed(42)` in `feature_engineering.py`). Seed change makes dataset non-deterministic.
- Bayesian shrinkage toward population priors (`shrink_rate`, `shrink_proportion` in `stats_utils.py`). Weight classes with <200 fights fall back to `"global"`.
- Elo uses variable K-factor (96/64/40/24 by experience bands) and >1 year inactivity decay.
- `CUTOFF_DATE = 2001-01-01` — fights before this are excluded from feature computation.

### Prediction
- All scripts average predictions from both orderings (A→B and B→A) to remove order-dependent bias.
- `predict_event.py` filters out fights **after** event date before building states — critical to prevent lookahead.
- `predict.py` uses `shap.TreeExplainer` on LightGBM base; others skip SHAP.
- `constrain_round_probas()` was removed along with the round model (predictions performed worse than the majority-class baseline).
- Fighters with 0 prior fights are **skipped**; <3 fights triggers a warning.

### Data Integrity
- `data/` and `models/` are generated artifacts — don't edit by hand.
- Feature column changes in `train_model.py` must be mirrored in all prediction scripts.
- `dataset.csv` includes `finish_type` (KO/SUB/DEC/OTHER) and `finish_round` (1-5) target columns.
