# UFC Fight Predictor

Pipeline de machine learning que predice el ganador de combates de UFC usando un **stacking ensemble** (LightGBM + XGBoost + Regresión Logística) entrenado sobre datos scrapeados de la web oficial de la UFC.

## Pipeline

El proyecto procesa los datos en orden cronológico (sin usar información del futuro), desde combates el `2012-01-01`.

```bash
python src/scraping/build_events_index.py     # 1. índice de eventos → data/events_index.json (omite eventos no finalizados)
python src/scraping/scrape_ufc.py             # 2. scrape de combates → data/fights.json, data/fighters_cache.json
python src/feature_engineering.py             # 3. features → data/dataset.csv
python src/train_model.py                     # 4. entrena el modelo
python src/prediction/predict.py              # 5. predicción interactiva (SHAP)
python src/prediction/predict_event.py --event "UFC 328: ..."   # 6. predicción de un evento
python src/prediction/backtest.py --start 2015-01-01            # 7. backtest sin lookahead
python src/prediction/predict_url.py "URL de evento de ufcstats" # 8. scrape + predicción de un evento
```

## Instalación

```bash
python -m venv .venv
.venv/bin/activate               # Linux/macOS
.venv/Scripts/activate           # Windows
pip install -r requirements.txt
playwright install chromium      # requerido por el scraper (pasos 1 y 2)
```

Todos los scripts se ejecutan desde la raíz del repositorio.

## Predicción

Todos los scripts de predicción promedian los resultados de ambos órdenes (A→B y B→A) para eliminar sesgo de orden.

### Interactiva (`predict.py`)

CLI interactiva con explicabilidad **SHAP**.

```bash
python src/prediction/predict.py --model models/ufc_stacking_ensemble.pkl --features models/ufc_stacking_ensemble_meta.pkl
```

### Por evento (`predict_event.py`)

Genera un JSON con las probabilidades de ganador para cada combate de un evento.

```bash
python src/prediction/predict_event.py --event "UFC 328: ..."
```

- `--exact`: fuerza coincidencia exacta del nombre del evento.
- `--model-path` / `--features-path`: rutas personalizadas a artefactos.

Se filtran los combates posteriores a la fecha del evento antes de construir los estados — clave para evitar *lookahead*.

### Backtest (`backtest.py`)

Evalúa el modelo sobre un período histórico sin *lookahead*: por cada combate el modelo solo ve estados y priors construidos con combates anteriores.

```bash
python src/prediction/backtest.py --start 2015-01-01 [--end 2019-12-31]
```

- Omite peleas debut (detectadas por `debut_date` en el cache, con fallback a 0 peleas previas) y empates/sin resultado.
- Reporta accuracy, AUC, log-loss y calibración por año.
- `--model-path` / `--features-path`: rutas personalizadas a artefactos.

### Desde URL (`predict_url.py`)

Scrapea una página de evento de ufcstats (Playwright) y predice todos sus combates:

```bash
python src/prediction/predict_url.py "http://ufcstats.com/event-details/..."
```

- Muestra una tabla con probabilidades, rounds programados y el récord en UFC (W-L-D) de cada luchador.
- Se omiten combates cuyos luchadores no estén en `data/fighters_cache.json` o con 0 peleas previas.
- Luchadores con menos de 3 peleas previas se marcan con `*`.
- Rounds: 5 para peleas titular (ícono de cinturón) y la primera pelea de la página; 3 en el resto.
- El URL se puede pasar posicional o con `--url`; `--model-path` / `--features-path` para rutas personalizadas.

## Arquitectura del modelo

`ChronologicalStackingEnsemble` (`src/ensemble_utils.py`):

- **Base:** LightGBM + XGBoost + LogisticRegression (Pipeline: imputer → scaler → LR) con un meta-LR.
- **Validación**: `TimeSeriesSplit(n_splits=5)` OOF.
- **Metadata**: guarda `raw_feature_cols`, `cat_cols`, `numeric_cols`, `feature_cols_final`, `model_type="stacking"`.
- **Target**: binario — probabilidad de que el luchador A gane.
- **Calibración**: los calibradores (`isotonic` y `platt`) se ajustan sobre las probabilidades *out-of-fold* del stacking vía un `TimeSeriesSplit` anidado (sin *lookahead*); el mejor por log-loss en test se guarda en `model.calibrator_name`. `predict_proba` devuelve probabilidades calibradas; `predict` sigue umbralizando en 0.5.

### Features e ingeniería

- Combos procesados **cronológicamente** sin *lookahead*; los priors se acumulan incrementalmente.
- Lados A/B se asignan **al azar** por combate (`random.seed(42)`).
- *Bayesian shrinkage* hacia priors poblacionales; clases de peso con < 200 combos caen a `"global"`.
- Elo con K-factor variable (96/64/40/24 por bandas de experiencia) y decaimiento por inactividad > 1 año.