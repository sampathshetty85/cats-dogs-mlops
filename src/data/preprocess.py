import os
import sys
import random

from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import IMAGE_SIZE, IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD, CLASSES  # noqa: E402
from src.report_writer import write_report  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "PetImages")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_valid_images(class_dir: str) -> tuple[list, int]:
    valid = []
    corrupt = 0
    for fname in sorted(os.listdir(class_dir)):
        fpath = os.path.join(class_dir, fname)
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue
        try:
            with Image.open(fpath) as img:
                img.verify()
            valid.append(fpath)
        except Exception:
            corrupt += 1
    return valid, corrupt


def split_files(files: list, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def resize_and_save(src_path: str, dst_path: str):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=95)


def main():
    corrupt_counts = {}
    split_counts = {split: {cls: 0 for cls in CLASSES.values()} for split in SPLITS}

    for raw_cls, out_cls in CLASSES.items():
        class_dir = os.path.join(RAW_DIR, raw_cls)
        print(f"[preprocess] Processing {raw_cls} images from {class_dir}")

        valid_files, corrupt = load_valid_images(class_dir)
        corrupt_counts[raw_cls] = corrupt
        print(f"[preprocess]   Valid: {len(valid_files):,}  Corrupt/skipped: {corrupt}")

        splits = split_files(valid_files)

        for split, files in splits.items():
            dst_dir = os.path.join(PROCESSED_DIR, split, out_cls)
            os.makedirs(dst_dir, exist_ok=True)
            for src in files:
                fname = os.path.basename(src)
                dst = os.path.join(dst_dir, fname)
                resize_and_save(src, dst)
            split_counts[split][out_cls] = len(files)
            print(f"[preprocess]   {split:5s}/{out_cls}: {len(files):,} images")

    print("\n[preprocess] Split summary:")
    for split in SPLITS:
        total = sum(split_counts[split].values())
        print(f"  {split:5s}: cat={split_counts[split]['cat']:,}  dog={split_counts[split]['dog']:,}  total={total:,}")

    report_lines = [
        "Dataset source : data/raw/PetImages/",
        "Output dir     : data/processed/",
        "Image size     : 224x224 RGB (LANCZOS resampling)",
        "Split ratio    : 80% train / 10% val / 10% test (stratified, seed=42)",
        "",
        "Corrupt files skipped:",
    ]
    for cls, count in corrupt_counts.items():
        report_lines.append(f"  {cls}: {count}")

    report_lines.append("")
    report_lines.append("Split counts:")
    for split in SPLITS:
        cat = split_counts[split]["cat"]
        dog = split_counts[split]["dog"]
        report_lines.append(f"  {split:5s}: cat={cat:,}  dog={dog:,}  total={cat+dog:,}")

    report_lines += [
        "",
        "TRAIN_TRANSFORMS: Resize(224) → RandomHorizontalFlip → RandomRotation(10)"
        " → ColorJitter(0.2,0.2) → ToTensor → Normalize(ImageNet)",
        "EVAL_TRANSFORMS : Resize(224) → ToTensor → Normalize(ImageNet)",
    ]

    write_report("step_2_preprocess", report_lines)


if __name__ == "__main__":
    main()
