# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python venv lives at `~/.venvs/cats-dogs-mlops/`. Always activate before running anything:

```bash
source ~/.venvs/cats-dogs-mlops/bin/activate
```

## Common Commands

```bash
# Run a single pipeline script
python src/data/download.py
python src/data/preprocess.py
python src/model/train.py

# Run full DVC pipeline (preprocess → train)
dvc repro

# Check pipeline/data state
dvc status
dvc dag

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_inference.py -v

# Lint
flake8 src/ tests/ --max-line-length=120

# Start API locally
uvicorn src.api.main:app --reload

# Start full monitoring stack (api + prometheus + grafana)
docker compose up -d

# View MLflow experiment runs
mlflow ui   # open http://127.0.0.1:5000

# Generate PDF report
python docs/generate_report.py

# Batch eval against live API
python scripts/batch_eval.py

# Smoke test against running stack
bash scripts/smoke_test.sh
```

## Architecture

### Pipeline flow (developer, run once)

```
src/data/download.py   → data/raw/PetImages/{Cat,Dog}/   (Kaggle, ~25k images)
src/data/preprocess.py → data/processed/{train,val,test}/{cat,dog}/  (224×224, 80/10/10)
src/model/train.py     → models/cats_dogs_cnn.pt  +  MLflow run
dvc add + dvc push     → DVC pointer files committed, binaries in /tmp/dvc-remote
```

The Docker image is built **after** training and bakes in the `.pt` file — no retraining inside Docker. Evaluators pull `ghcr.io/sampathshetty85/cats-dogs-mlops:latest` and run `docker compose up -d`.

### Key modules

- **`src/report_writer.py`** — `write_report(step_name, lines)` writes a timestamped `.txt` to `output/`. Every pipeline script must call this at the end; all output reports are committed to git as pipeline evidence.
- **`src/data/preprocess.py`** — exports `TRAIN_TRANSFORMS` and `EVAL_TRANSFORMS` as module-level constants. Both `train.py` and the test suite import them from here — do not redefine them elsewhere.
- **`src/model/architecture.py`** — defines `SimpleCNN`, a custom CNN (not a pretrained backbone). Input: `(B, 3, 224, 224)`, output: `(B, 1)` sigmoid probability. Label threshold: 0.5 (≥ 0.5 → dog, < 0.5 → cat).
- **`src/api/main.py`** — FastAPI app. Model is loaded once at startup via lifespan context manager and shared across requests. Three endpoints: `/health`, `/predict` (multipart file upload), `/metrics` (Prometheus).

### Data & model versioning

`data/raw/`, `data/processed/`, and `models/*.pt` are gitignored and DVC-tracked. Their pointer files (`data/raw.dvc`, `data/processed.dvc`, `models/cats_dogs_cnn.pt.dvc`) are what gets committed. DVC remote is a local folder at `/tmp/dvc-remote`.

### CI/CD

`.github/workflows/ci.yml` — lint + test on every push/PR to `main`, then build and push Docker image to `ghcr.io` (only on `main`).
`.github/workflows/cd.yml` — triggers on push to `main` and `workflow_dispatch`; pulls new image and runs `scripts/smoke_test.sh`.

### Monitoring

Prometheus scrapes `api:8000/metrics` every 15s. Grafana auto-provisions the datasource and `monitoring/grafana/dashboards/cats_dogs.json` dashboard. Custom counter: `cats_dogs_predictions_total{prediction_label="cat|dog"}`.
