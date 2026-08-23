import os
import sys
import random
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.report_writer import write_report  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.join(REPO_ROOT, "data", "processed", "test")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
N_PER_CLASS = 25
SEED = 42


def collect_images(class_name: str, n: int) -> list[tuple[str, str]]:
    class_dir = os.path.join(TEST_DIR, class_name)
    files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rng = random.Random(SEED)
    selected = rng.sample(files, min(n, len(files)))
    return [(os.path.join(class_dir, f), class_name) for f in selected]


def predict(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()


def main():
    import mlflow

    print(f"[batch_eval] API: {API_URL}")
    print(f"[batch_eval] Collecting {N_PER_CLASS} cat + {N_PER_CLASS} dog images from test set...")

    samples = collect_images("cat", N_PER_CLASS) + collect_images("dog", N_PER_CLASS)
    random.Random(SEED).shuffle(samples)

    results = []
    for i, (path, true_label) in enumerate(samples, 1):
        try:
            pred = predict(path)
            predicted_label = pred["label"]
            results.append({
                "true": true_label,
                "predicted": predicted_label,
                "correct": true_label == predicted_label,
                "probability": pred["probability"],
                "latency_ms": pred["inference_time_ms"],
            })
            if i % 10 == 0:
                print(f"[batch_eval] {i}/{len(samples)} done")
        except Exception as e:
            print(f"[batch_eval] WARN: failed on {path}: {e}")

    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total else 0.0

    tp = sum(1 for r in results if r["true"] == "cat" and r["predicted"] == "cat")
    fp = sum(1 for r in results if r["true"] == "dog" and r["predicted"] == "cat")
    fn = sum(1 for r in results if r["true"] == "cat" and r["predicted"] == "dog")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0.0

    print(f"\n[batch_eval] Results ({total} images):")
    print(f"  Accuracy  : {accuracy:.4f} ({correct}/{total} correct)")
    print(f"  Precision : {precision:.4f}  (cat class)")
    print(f"  Recall    : {recall:.4f}  (cat class)")
    print(f"  Avg latency: {avg_latency:.1f} ms")

    mlflow.set_tracking_uri(f"file://{os.path.join(REPO_ROOT, 'mlruns')}")
    mlflow.set_experiment("cats-dogs-classification")
    with mlflow.start_run(run_name="batch_eval_post_deploy"):
        mlflow.set_tag("stage", "post_deploy")
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision_cat": precision,
            "recall_cat": recall,
            "avg_latency_ms": avg_latency,
            "total_samples": total,
        })
        print(f"[batch_eval] Results logged to MLflow.")

    report_lines = [
        f"API URL      : {API_URL}",
        f"Total samples: {total} ({N_PER_CLASS} cat + {N_PER_CLASS} dog from test set)",
        f"Correct      : {correct}",
        f"Accuracy     : {accuracy:.4f}",
        f"Precision    : {precision:.4f}  (cat class)",
        f"Recall       : {recall:.4f}  (cat class)",
        f"Avg latency  : {avg_latency:.1f} ms",
        "",
        "MLflow run   : batch_eval_post_deploy (tag: stage=post_deploy)",
    ]
    write_report("step_11_monitoring", report_lines)
    print("[batch_eval] Done.")


if __name__ == "__main__":
    main()
