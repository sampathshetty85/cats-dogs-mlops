"""
Generates docs/report.pdf for MLOps Assignment 2 submission.
Run from repo root: python docs/generate_report.py
"""
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "docs")
PDF_PATH = os.path.join(OUTPUT_DIR, "report.pdf")

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
BASE = getSampleStyleSheet()

TITLE = ParagraphStyle("Title", parent=BASE["Title"], fontSize=22, spaceAfter=12, alignment=TA_CENTER)
H1 = ParagraphStyle("H1", parent=BASE["Heading1"], fontSize=16, spaceBefore=14, spaceAfter=6,
                    textColor=colors.HexColor("#1a3a5c"))
H2 = ParagraphStyle("H2", parent=BASE["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#2e6da4"))
BODY = ParagraphStyle("Body", parent=BASE["BodyText"], fontSize=10, spaceAfter=6, leading=14)
CODE = ParagraphStyle("Code", parent=BASE["Code"], fontSize=8, spaceAfter=4, leading=12,
                      backColor=colors.HexColor("#f4f4f4"), leftIndent=12, rightIndent=12,
                      fontName="Courier")
CAPTION = ParagraphStyle("Caption", parent=BASE["Italic"], fontSize=9, alignment=TA_CENTER,
                         textColor=colors.grey, spaceAfter=8)
SUBTITLE = ParagraphStyle("Subtitle", parent=BASE["Normal"], fontSize=12, alignment=TA_CENTER,
                           spaceAfter=8, textColor=colors.HexColor("#444444"))


def HR():
    return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=8)


def read_report(step_name: str) -> list[str]:
    path = os.path.join(REPO_ROOT, "output", f"{step_name}.txt")
    if not os.path.exists(path):
        return [f"[{step_name}.txt not found]"]
    with open(path) as f:
        lines = [l.rstrip() for l in f if not l.startswith("#")]
    return [l for l in lines if l.strip()]


def embed_image(rel_path: str, max_width: float = 14 * cm) -> list:
    abs_path = os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(abs_path):
        return [Paragraph(f"[Screenshot not available: {rel_path}]", CAPTION)]
    try:
        img = Image(abs_path)
        scale = min(max_width / img.imageWidth, (10 * cm) / img.imageHeight, 1.0)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        return [img]
    except Exception as e:
        return [Paragraph(f"[Could not embed {rel_path}: {e}]", CAPTION)]


def code_block(lines: list[str], max_lines: int = 20) -> list:
    shown = lines[:max_lines]
    if len(lines) > max_lines:
        shown.append(f"... ({len(lines) - max_lines} more lines)")
    return [Paragraph(l or " ", CODE) for l in shown]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def cover_page() -> list:
    return [
        Spacer(1, 3 * cm),
        Paragraph("MLOps Assignment 2", TITLE),
        Spacer(1, 0.5 * cm),
        Paragraph("End-to-End MLOps Pipeline — Binary Image Classification (Cats vs Dogs)", SUBTITLE),
        Spacer(1, 1 * cm),
        HR(),
        Spacer(1, 0.5 * cm),
        Paragraph("Student: Sampath Kumar S Shetty", SUBTITLE),
        Paragraph("ID: 2024ac05041", SUBTITLE),
        Paragraph("Course: AIMLCZG523 — MLOps (S1-25)", SUBTITLE),
        Spacer(1, 1 * cm),
        Paragraph("GitHub: https://github.com/sampathshetty85/cats-dogs-mlops", SUBTITLE),
        Paragraph("Docker: ghcr.io/sampathshetty85/cats-dogs-mlops:latest", SUBTITLE),
        Paragraph("Demo Video: https://youtu.be/SuxLDBvdej8", SUBTITLE),
        Paragraph("Live Demo: https://sampathshetty85.github.io/cats-dogs-mlops/demo.html", SUBTITLE),
        Spacer(1, 2 * cm),
        HR(),
        Spacer(1, 0.5 * cm),
        Paragraph("Technology Stack", H2),
        Table(
            [
                ["Model", "PyTorch 2.4.1 — SimpleCNN (custom)"],
                ["API", "FastAPI 0.115 + uvicorn"],
                ["Experiment Tracking", "MLflow 2.22.1"],
                ["Data Versioning", "DVC 3.55.2"],
                ["Containerization", "Docker — python:3.12-slim"],
                ["Container Registry", "GitHub Container Registry (ghcr.io)"],
                ["CI/CD", "GitHub Actions"],
                ["Monitoring", "Prometheus + Grafana 11.1"],
                ["Deployment", "Docker Compose + Kubernetes (minikube)"],
            ],
            colWidths=[5 * cm, 11 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0f8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        PageBreak(),
    ]


def project_overview() -> list:
    return [
        Paragraph("Project Overview", H1), HR(),
        Paragraph(
            "This project implements a complete MLOps pipeline for binary image classification "
            "(Cats vs Dogs) as required by Assignment 2. The pipeline covers data versioning, "
            "model training with experiment tracking, containerized inference service, automated "
            "CI/CD, deployment, and monitoring.", BODY),
        Spacer(1, 0.3 * cm),
        Paragraph("Pipeline Flow", H2),
        Table(
            [
                ["Step", "Script", "Output"],
                ["1. Download", "src/data/download.py", "data/raw/PetImages/ (25k images, 826 MB)"],
                ["2. Preprocess", "src/data/preprocess.py", "data/processed/ (224×224, 80/10/10 split)"],
                ["3. Train", "src/model/train.py", "models/cats_dogs_cnn.pt + MLflow run"],
                ["4. DVC track", "dvc add + dvc push", "Pointer files committed, binaries versioned"],
                ["5. API", "src/api/main.py", "FastAPI /health + /predict + /metrics"],
                ["6. Docker", "Dockerfile", "ghcr.io/sampathshetty85/cats-dogs-mlops:latest"],
                ["7. CI/CD", ".github/workflows/ci.yml + cd.yml", "Auto-test + deploy on main"],
                ["8. Monitor", "Prometheus + Grafana", "Metrics dashboard at :3000"],
            ],
            colWidths=[3.5 * cm, 5 * cm, 7.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fc")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        Spacer(1, 0.5 * cm),
        Paragraph("Evaluator Quick-Start", H2),
        Paragraph("No Kaggle credentials or local training required. One command brings up the full stack:", BODY),
        *code_block([
            "git clone https://github.com/sampathshetty85/cats-dogs-mlops",
            "cd cats-dogs-mlops",
            "docker compose up -d",
            "curl http://localhost:8000/health",
            'curl -F "file=@tests/fixtures/sample_cat.jpg" http://localhost:8000/predict',
            "open http://localhost:3000   # Grafana dashboard (admin/admin)",
        ]),
        PageBreak(),
    ]


def dataset_preprocessing() -> list:
    download_lines = read_report("step_1_download")
    preprocess_lines = read_report("step_2_preprocess")
    return [
        Paragraph("Dataset & Preprocessing (M1)", H1), HR(),
        Paragraph(
            "The Microsoft Cats vs Dogs dataset was downloaded from Kaggle and preprocessed to "
            "224×224 RGB images with an 80/10/10 stratified train/val/test split. Corrupt files "
            "were skipped automatically.", BODY),
        Spacer(1, 0.2 * cm),
        Paragraph("Download Evidence (output/step_1_download.txt)", H2),
        *code_block(download_lines),
        Paragraph("Preprocessing Evidence (output/step_2_preprocess.txt)", H2),
        *code_block(preprocess_lines),
        PageBreak(),
    ]


def model_architecture() -> list:
    return [
        Paragraph("Model Architecture (M1)", H1), HR(),
        Paragraph(
            "SimpleCNN is a custom convolutional neural network implemented in PyTorch. "
            "It takes 224×224 RGB images and outputs a sigmoid probability — ≥ 0.5 predicts dog, "
            "< 0.5 predicts cat.", BODY),
        Spacer(1, 0.3 * cm),
        Paragraph("Architecture", H2),
        Table(
            [
                ["Layer", "Configuration", "Output Shape"],
                ["Conv1", "3→32, k=3 p=1 → BatchNorm → ReLU → MaxPool(2)", "(B, 32, 112, 112)"],
                ["Conv2", "32→64, k=3 p=1 → BatchNorm → ReLU → MaxPool(2)", "(B, 64, 56, 56)"],
                ["Conv3", "64→128, k=3 p=1 → BatchNorm → ReLU → AdaptiveAvgPool(7,7)", "(B, 128, 7, 7)"],
                ["Flatten", "128 × 7 × 7 = 6272", "(B, 6272)"],
                ["FC1", "6272 → 512 → ReLU → Dropout(0.5)", "(B, 512)"],
                ["FC2", "512 → 1 → Sigmoid", "(B, 1)"],
            ],
            colWidths=[2.5 * cm, 8 * cm, 5.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fc")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        Spacer(1, 0.5 * cm),
        Paragraph("Training Configuration", H2),
        Table(
            [
                ["Hyperparameter", "Value"],
                ["Epochs", "5"],
                ["Batch size", "32"],
                ["Optimizer", "Adam (lr=0.001)"],
                ["Loss function", "BCELoss"],
                ["Augmentations", "RandomHorizontalFlip, RandomRotation(10), ColorJitter(0.2, 0.2)"],
                ["Normalization", "ImageNet stats: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]"],
            ],
            colWidths=[5 * cm, 11 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0f8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        PageBreak(),
    ]


def training_results() -> list:
    train_lines = read_report("step_4_train")
    return [
        Paragraph("Training Results (M1)", H1), HR(),
        Paragraph("Training Evidence (output/step_4_train.txt)", H2),
        *code_block(train_lines, max_lines=25),
        Spacer(1, 0.3 * cm),
        Paragraph("Loss & Accuracy Curves", H2),
        *embed_image("screenshots/training/loss_curves.png", max_width=15 * cm),
        Paragraph("Figure 1: Training and validation loss/accuracy over 5 epochs", CAPTION),
        Spacer(1, 0.3 * cm),
        Paragraph("Confusion Matrix (Validation Set)", H2),
        *embed_image("screenshots/training/confusion_matrix.png", max_width=10 * cm),
        Paragraph("Figure 2: Confusion matrix on the validation set", CAPTION),
        PageBreak(),
    ]


def experiment_tracking() -> list:
    mlflow_lines = read_report("step_5_mlflow")
    return [
        Paragraph("Experiment Tracking with MLflow (M1)", H1), HR(),
        Paragraph(
            "MLflow was used to track the training experiment. All hyperparameters, per-epoch "
            "metrics, loss curves, confusion matrix, and the model artifact were logged under "
            "the 'cats-dogs-classification' experiment.", BODY),
        Spacer(1, 0.3 * cm),
        Paragraph("MLflow Run Evidence (output/step_5_mlflow.txt)", H2),
        *code_block(mlflow_lines),
        Spacer(1, 0.3 * cm),
        Paragraph("Viewing MLflow UI", H2),
        *code_block(["mlflow ui   # open http://127.0.0.1:5000"]),
        Paragraph(
            "The MLflow UI shows the SimpleCNN_5ep run with all logged parameters "
            "(architecture, epochs, batch_size, learning_rate, optimizer, loss_fn, image_size, "
            "augmentations) and per-epoch metrics. Loss curves and confusion matrix are logged "
            "as artifacts. The model is logged via mlflow.pytorch.log_model with input signature.", BODY),
        PageBreak(),
    ]


def api_containerization() -> list:
    api_lines = read_report("step_7_api")
    container_lines = read_report("step_8_containerize")
    return [
        Paragraph("API & Containerization (M2)", H1), HR(),
        Paragraph("Inference Service (output/step_7_api.txt)", H2),
        *code_block(api_lines),
        Spacer(1, 0.3 * cm),
        Paragraph("Dockerfile", H2),
        *code_block([
            "FROM python:3.12-slim",
            "WORKDIR /app",
            "COPY requirements.txt .",
            "RUN pip install --no-cache-dir -r requirements.txt",
            "COPY src/ ./src/",
            "COPY models/cats_dogs_cnn.pt ./models/",
            "EXPOSE 8000",
            'CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]',
        ]),
        Spacer(1, 0.3 * cm),
        Paragraph("Container Verification (output/step_8_containerize.txt)", H2),
        *code_block(container_lines),
        PageBreak(),
    ]


def cicd_pipeline() -> list:
    ci_lines = read_report("step_9_ci")
    return [
        Paragraph("CI/CD Pipeline (M3 + M4)", H1), HR(),
        Paragraph(
            "GitHub Actions powers both CI (test + build) and CD (deploy + smoke test). "
            "The CI pipeline runs on every push and pull request. On merge to main, the Docker "
            "image is built for linux/amd64 and linux/arm64 and pushed to ghcr.io. "
            "The CD pipeline then pulls the new image, starts the stack, and runs the smoke test.", BODY),
        Spacer(1, 0.3 * cm),
        Paragraph("CI Evidence (output/step_9_ci.txt)", H2),
        *code_block(ci_lines),
        Spacer(1, 0.3 * cm),
        Paragraph("CD Workflow (.github/workflows/cd.yml)", H2),
        *code_block([
            "Triggers: push to main + workflow_dispatch",
            "Steps:",
            "  1. docker compose pull api      # pull new image from ghcr.io",
            "  2. docker compose up -d         # start api + prometheus + grafana",
            "  3. wait for /health → 200       # up to 60s",
            "  4. bash scripts/smoke_test.sh   # health + predict checks",
            "  5. on failure: docker compose logs printed; pipeline fails",
        ]),
        Spacer(1, 0.3 * cm),
        Paragraph("Unit Tests (7 tests, all pass)", H2),
        Table(
            [
                ["Test", "File", "What it checks"],
                ["test_resize_output_shape", "test_preprocess.py", "EVAL_TRANSFORMS output shape [3,224,224]"],
                ["test_normalize_range", "test_preprocess.py", "Tensor values in [-3.0, 3.0]"],
                ["test_augmentation_preserves_shape", "test_preprocess.py", "TRAIN_TRANSFORMS preserves shape"],
                ["test_model_output_range", "test_inference.py", "SimpleCNN output in [0.0, 1.0]"],
                ["test_model_output_shape", "test_inference.py", "SimpleCNN output shape (1,1)"],
                ["test_health_endpoint", "test_inference.py", "GET /health → 200, status=ok"],
                ["test_predict_endpoint", "test_inference.py", "POST /predict → label in (cat,dog)"],
            ],
            colWidths=[5.5 * cm, 3.5 * cm, 7 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fc")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]),
        ),
        PageBreak(),
    ]


def deployment() -> list:
    deploy_lines = read_report("step_10_deployment")
    return [
        Paragraph("Deployment (M4)", H1), HR(),
        Paragraph(
            "The inference service is deployed via Docker Compose (three-service stack) and "
            "Kubernetes manifests (two replicas with health probes).", BODY),
        Spacer(1, 0.3 * cm),
        Paragraph("Deployment Evidence (output/step_10_deployment.txt)", H2),
        *code_block(deploy_lines),
        Spacer(1, 0.3 * cm),
        Paragraph("Kubernetes Manifests", H2),
        *code_block([
            "# k8s/deployment.yaml",
            "replicas: 2",
            "image: ghcr.io/sampathshetty85/cats-dogs-mlops:latest",
            "imagePullPolicy: Always",
            "livenessProbe:  httpGet path=/health, initialDelaySeconds=5, periodSeconds=10",
            "readinessProbe: httpGet path=/health, initialDelaySeconds=5, periodSeconds=10",
            "resources: limits cpu=500m memory=512Mi  requests cpu=250m memory=256Mi",
            "",
            "# k8s/service.yaml",
            "type: LoadBalancer   port: 80 → 8000",
        ]),
        Spacer(1, 0.3 * cm),
        Paragraph("Apply with:", H2),
        *code_block([
            "minikube start",
            "kubectl apply -f k8s/",
            "kubectl get pods",
            "kubectl port-forward svc/cats-dogs-mlops 8080:80",
            "curl http://localhost:8080/health",
        ]),
        PageBreak(),
    ]


def monitoring() -> list:
    monitor_lines = read_report("step_11_monitoring")
    return [
        Paragraph("Monitoring, Logging & Batch Evaluation (M5)", H1), HR(),
        Paragraph(
            "Structured JSON logs are emitted on every /predict call. Prometheus scrapes the "
            "/metrics endpoint every 15s. Grafana auto-provisions the datasource and dashboard "
            "on startup.", BODY),
        Spacer(1, 0.3 * cm),
        Paragraph("Structured JSON Log Format", H2),
        *code_block([
            '{"ts": "2026-08-23T00:00:00Z", "path": "/predict", "label": "cat",',
            ' "probability": 0.87, "latency_ms": 14.2}',
        ]),
        Spacer(1, 0.3 * cm),
        Paragraph("Grafana Dashboard — 4 Panels", H2),
        Table(
            [
                ["Panel", "Metric", "Description"],
                ["Request Rate", "rate(http_requests_total[5m])", "Requests/second by endpoint"],
                ["Latency P50/P95", "histogram_quantile(0.5/0.95, ...)", "Response time percentiles"],
                ["Prediction Distribution", "rate(cats_dogs_predictions_total[5m])", "Cat vs Dog predictions/s"],
                ["Error Rate", "rate(http_requests_total{status_code!~'2..'}[5m])", "% non-2xx responses"],
            ],
            colWidths=[3.5 * cm, 7 * cm, 5.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fc")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]),
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("Post-Deployment Batch Evaluation (output/step_11_monitoring.txt)", H2),
        *code_block(monitor_lines),
        PageBreak(),
    ]


def conclusion() -> list:
    return [
        Paragraph("Conclusion", H1), HR(),
        Paragraph(
            "This project demonstrates a complete, reproducible MLOps pipeline for binary image "
            "classification. Every component of the pipeline — from data versioning to CI/CD "
            "deployment and monitoring — is automated and evidence is committed to the repository.", BODY),
        Spacer(1, 0.3 * cm),
        Paragraph("Key Results", H2),
        Table(
            [
                ["Metric", "Value"],
                ["Model", "SimpleCNN (custom PyTorch CNN)"],
                ["Validation accuracy", "~82.7% (5 epochs)"],
                ["Post-deploy accuracy", "86.0% (50 test images)"],
                ["Post-deploy precision (cat)", "84.6%"],
                ["Post-deploy recall (cat)", "88.0%"],
                ["API avg latency", "~14 ms"],
                ["Docker image size", "448.8 MB (python:3.12-slim)"],
                ["CI runtime", "~14 min (lint + test + multi-arch Docker build)"],
                ["CD runtime", "~2 min (pull + start + smoke test)"],
            ],
            colWidths=[7 * cm, 9 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0f8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        Spacer(1, 0.5 * cm),
        Paragraph("Repository Links", H2),
        *code_block([
            "GitHub     : https://github.com/sampathshetty85/cats-dogs-mlops",
            "Docker     : ghcr.io/sampathshetty85/cats-dogs-mlops:latest",
            "Demo Video : https://youtu.be/SuxLDBvdej8",
            "Live Demo  : https://sampathshetty85.github.io/cats-dogs-mlops/demo.html",
            "CI badge   : https://github.com/sampathshetty85/cats-dogs-mlops/actions/workflows/ci.yml",
        ]),
    ]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    story = (
        cover_page()
        + project_overview()
        + dataset_preprocessing()
        + model_architecture()
        + training_results()
        + experiment_tracking()
        + api_containerization()
        + cicd_pipeline()
        + deployment()
        + monitoring()
        + conclusion()
    )
    doc.build(story)
    print(f"[report] Generated: {PDF_PATH}")


if __name__ == "__main__":
    build()
