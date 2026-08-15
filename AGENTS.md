# AGENTS.md

## Pipeline (sequential)
1. `python src/scraping/build_events_index.py` → `data/events_index.json` (skips upcoming/unfinished events)
2. `python src/scraping/scrape_ufc.py` → `data/fights.json`, `data/fighters_cache.json`
3. `python src/feature_engineering.py` → `data/dataset.csv`
4. `python src/train_model.py` → `models/ufc_stacking_ensemble.pkl` + `_meta.pkl` + feature importance PNG (run manually; also fits isotonic/Platt probability calibrators on nested-CV OOF and picks the best by test Brier — slow)
5. `python src/prediction/predict.py` — interactive CLI (SHAP). Args: `--model`, `--features`.
6. `python src/prediction/predict_event.py --event "UFC 328: ..."` — event JSON with winner probabilities. Args: `--exact`, `--model-path`, `--features-path`.
7. `python src/prediction/predict_batch.py "FighterA,FighterB,Category,5" "FighterC,FighterD"` — batch table. Each arg: `F1,F2[,WeightClass[,Rounds]]`. Rounds defaults 3, WeightClass defaults "Catch Weight". No model-path CLI flags (hardcoded paths). Quirk: in the 3-field form `F1,F2,X`, `X` is parsed as rounds if it's `"3"`/`"5"`, otherwise as weight class.
8. `python src/prediction/backtest.py --start 2015-01-01 [--end ...]` — no-lookahead backtest (accuracy/AUC/log-loss/calibration per year). Debut fights and draws/NCs are skipped. Args: `--model-path`, `--features-path`.

To run the pipeline (skip step 4, run it by hand):
```bash
source .venv/bin/activate   # or .venv/Scripts/activate on Windows
python src/scraping/build_events_index.py
python src/scraping/scrape_ufc.py
python src/feature_engineering.py
```

## Setup
- Activate the venv first — `.venv/bin/activate` on Linux/macOS, `.venv/Scripts/activate` on Windows. All scripts from repo root.
- Scraper needs `playwright install chromium` before first run (both `build_events_index.py` and `scrape_ufc.py` use Playwright async headless Chromium).
- Scrapers are resumable: `scrape_ufc.py` only processes events with `"scrapped": false` in `data/events_index.json`, flipping the flag per event; `build_events_index.py` preserves existing entries.
- Step 4 runs 50 random LightGBM + 20 random XGBoost hyperparameter trials (early stopping), then a nested-CV pass to calibrate probabilities — slow; run it manually and let it finish. Hyperparameter trials are selected by **validation log-loss** (a proper score), not AUC. Calibrators are fitted on out-of-fold stacking probabilities via an outer `TimeSeriesSplit(n_splits=5)` (no lookahead). Candidates are `isotonic`, `platt` (`PlattCalibrator`), and their capped variants (`CappedCalibrator` in `ensemble_utils.py`, which forbids mapping raw p<0.5 any higher — an underdog-inflation guard). The winner is chosen by **OOF-holdout log-loss** (a chronological holdout of the OOF points — never the test set, avoiding selection bias) and stored on the model (`model.calibrator_name`); OOF scores live in `model.calibrator_scores_`. Calibrators are fitted **and** selected on only the most recent 50% of the OOF points (`calib_recent_fraction`) — calibrating on the whole training era baked in stale pre-2020 underdog relationships and inflated underdog probabilities on recent fights. `predict_proba` returns calibrated probabilities; `predict` still thresholds at 0.5.
- Calibration/sharpness diagnostics (ECE, Brier decomposition, favorite/underdog split, sharpness) are printed by `train_model.py` and `backtest.py`. Use those to judge model quality; there is no dedicated model-comparison script.
- Model artifacts use Git LFS (`*.pkl filter=lfs` in `.gitattributes`). No linting, typechecking, or test harness.

## Architecture

### Model: Stacking Ensemble
`ChronologicalStackingEnsemble` (`src/ensemble_utils.py`) — **LightGBM + XGBoost + LogisticRegression** (Pipeline: imputer→scaler→LR) with a meta-LR, `TimeSeriesSplit(n_splits=5)` OOF. Metadata stores `raw_feature_cols`, `cat_cols`, `numeric_cols`, `feature_cols_final`, `model_type="stacking"`.

Single model, trained on the shared feature pipeline:
- **Winner** (binary): `ufc_stacking_ensemble.pkl` — prob fighter A wins (calibrated: see Setup note)

### Shared Modules
- `src/config.py` — paths and constants (`CUTOFF_DATE`, `ELO_K`, `ELO_INITIAL`, `WEIGHT_CLASSES`).
- `src/fighter_engine.py` — **single source of truth** for Elo, state tracking, feature computation, prediction rows, and `predict_fight()`.
- `src/stats_utils.py` — priors/shrinkage and composite-feature helpers.

`predict.py`, `predict_event.py`, `predict_batch.py`, and `feature_engineering.py` all import from these modules — no duplicated logic. If you change feature logic, do it in `fighter_engine.py`.

Layout: shared library modules and the `feature_engineering.py`/`train_model.py` pipeline scripts live at `src/` root; scraping scripts live in `src/scraping/` and prediction scripts in `src/prediction/`. The `prediction/` entry scripts add `src/` to `sys.path` with a small bootstrap (`sys.path.insert(0, ...parents[1])`) so they can `from config import ...` and so the pickled models (which reference `ensemble_utils`, `fighter_engine`, ...) keep loading.

### Feature Engineering
- Fights processed **chronologically** with no lookahead. Priors accumulate incrementally via `PriorAccumulator.add()`.
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
- Feature column changes in `train_model.py` must be mirrored in `fighter_engine.build_prediction_row()`.
- `dataset.csv` includes `finish_type` (KO/SUB/DEC/OTHER) and `finish_round` (1-5) target columns.
