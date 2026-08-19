# Cats vs Dogs MLOps Pipeline

**Student:** Sampath Kumar S Shetty | **ID:** 2024ac05041
**Course:** AIMLCZG523 — MLOps

[![CI](https://github.com/sampathshetty85/cats-dogs-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/sampathshetty85/cats-dogs-mlops/actions/workflows/ci.yml)

## Overview

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs). Built with PyTorch, FastAPI, DVC, MLflow, Docker, Kubernetes, Prometheus, and Grafana.

## Evaluator Quick-Start

No Kaggle credentials or local training required. The full stack comes up with one command:

```bash
git clone https://github.com/sampathshetty85/cats-dogs-mlops
cd cats-dogs-mlops
docker compose up -d
```

Then test:

```bash
curl http://localhost:8000/health
curl -F "file=@tests/fixtures/sample_cat.jpg" http://localhost:8000/predict
open http://localhost:3000   # Grafana dashboard (admin / admin)
```

## Project Structure

```
cats-dogs-mlops/
├── src/
│   ├── data/         # download.py, preprocess.py
│   ├── model/        # architecture.py, train.py, package.py
│   └── api/          # main.py, schemas.py
├── tests/            # pytest unit tests + fixtures
├── scripts/          # smoke_test.sh, batch_eval.py
├── notebooks/        # 01_eda.ipynb
├── monitoring/       # prometheus.yml, grafana dashboards
├── k8s/              # deployment.yaml, service.yaml
├── docs/             # report.pdf, GitHub Pages
├── output/           # validation reports (committed)
├── screenshots/      # evidence screenshots (committed)
├── dvc.yaml          # pipeline: preprocess → train
├── Dockerfile
└── docker-compose.yml
```

## Pipeline

```
python src/data/download.py       # Kaggle → data/raw/
python src/data/preprocess.py     # resize 224×224, 80/10/10 split
python src/model/train.py         # SimpleCNN, 5 epochs, MLflow logged
dvc add + dvc push                # version data and model
docker compose up -d              # evaluator: pull ghcr.io image + start stack
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Upload image → `{"label": "cat", "probability": 0.87, "inference_time_ms": 14.2}` |
| `/metrics` | GET | Prometheus metrics |

## Links

- GitHub: https://github.com/sampathshetty85/cats-dogs-mlops
- Docker image: `ghcr.io/sampathshetty85/cats-dogs-mlops:latest`
- MLflow: run `mlflow ui` from project root to view experiment runs
