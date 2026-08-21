import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.report_writer import write_report

KAGGLE_JSON = os.path.expanduser("~/.kaggle/kaggle.json")
DATASET = "shaunthesheep/microsoft-catsvsdogs-dataset"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
CLASSES = ["Cat", "Dog"]


def check_credentials():
    if not os.path.exists(KAGGLE_JSON):
        raise FileNotFoundError(
            f"Kaggle credentials not found at {KAGGLE_JSON}.\n"
            "Create one at https://www.kaggle.com/settings → Legacy API Credentials → Create Legacy API Key\n"
            f"Then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json {KAGGLE_JSON} && chmod 600 {KAGGLE_JSON}"
        )


def download_dataset(raw_dir: str):
    os.makedirs(raw_dir, exist_ok=True)
    print(f"[download] Downloading {DATASET} → {raw_dir}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", raw_dir, "--unzip"],
        check=True,
        text=True,
    )
    print(result.stdout or "")


def count_images(raw_dir: str) -> dict:
    counts = {}
    for cls in CLASSES:
        cls_dir = os.path.join(raw_dir, "PetImages", cls)
        if not os.path.isdir(cls_dir):
            counts[cls] = 0
            continue
        counts[cls] = sum(
            1 for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    return counts


def total_size_mb(raw_dir: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(raw_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 1)


def main():
    check_credentials()
    download_dataset(RAW_DIR)

    counts = count_images(RAW_DIR)
    total_images = sum(counts.values())
    size_mb = total_size_mb(RAW_DIR)
    abs_raw = os.path.abspath(RAW_DIR)

    print(f"[download] Cat images : {counts.get('Cat', 0):,}")
    print(f"[download] Dog images : {counts.get('Dog', 0):,}")
    print(f"[download] Total      : {total_images:,}")
    print(f"[download] Size on disk: {size_mb} MB")
    print(f"[download] Destination : {abs_raw}")

    report_lines = [
        f"Dataset      : {DATASET}",
        f"Source       : https://www.kaggle.com/datasets/{DATASET}",
        f"Destination  : {abs_raw}",
        f"Cat images   : {counts.get('Cat', 0):,}",
        f"Dog images   : {counts.get('Dog', 0):,}",
        f"Total images : {total_images:,}",
        f"Size on disk : {size_mb} MB",
        f"DVC remote   : /tmp/dvc-remote (local)",
    ]
    write_report("step_1_download", report_lines)


if __name__ == "__main__":
    main()
