# AGENTS.md

## Project Overview
This repository is a UFC fight prediction pipeline. The core flow is: build event indexes, scrape fight data, engineer features, train a model, then run interactive or event-level predictions.

## Working Conventions
- Run scripts from the repository root so relative paths like `data/...` and `models/...` resolve correctly.
- Treat files under `data/` and `models/` as generated artifacts unless a task explicitly asks to change the pipeline that produces them.
- Keep training and prediction feature schemas synchronized. If feature columns change in `src/train_model.py`, update the consumers in `src/predict.py` and `src/predict_event.py` together.
- Preserve the chronological, no-lookahead setup in feature engineering and evaluation.
- Scraping is browser-driven and may require realistic headers, timing, or retry handling if the site blocks requests.

## Main Scripts
- `src/build_events_index.py`: builds `data/events_index.json`.
- `src/scrape_ufc.py`: scrapes UFC data into `data/fights.json` and `data/fighters_cache.json`.
- `src/feature_engineering.py`: creates `data/dataset.csv` from historical fights.
- `src/train_model.py`: trains and saves the model artifacts in `models/`.
- `src/predict.py`: interactive fighter-vs-fighter CLI.
- `src/predict_event.py`: predicts an entire event from `--event`.

## Validation
- Prefer narrow script runs over broad repo-wide checks.
- Validate changes to the pipeline by re-running the smallest affected script, then confirm downstream scripts still load the saved artifacts.
- There is no formal test harness in the repo, so keep checks targeted and reproducible.

## Useful Reference
- See [mejoras.md](mejoras.md) for the model-improvement roadmap and known future work.
