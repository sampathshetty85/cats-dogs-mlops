import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.pytorch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.model.architecture import SimpleCNN
from src.data.preprocess import TRAIN_TRANSFORMS, EVAL_TRANSFORMS
from src.report_writer import write_report

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "cats_dogs_cnn.pt")
SCREENSHOTS_DIR = os.path.join(REPO_ROOT, "screenshots", "training")

EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders():
    train_ds = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "train"), transform=TRAIN_TRANSFORMS)
    val_ds = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "val"), transform=EVAL_TRANSFORMS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.class_to_idx


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def save_curves(train_losses, val_losses, train_accs, val_accs):
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plt.style.use("seaborn-v0_8-whitegrid")
    epochs = range(1, EPOCHS + 1)

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Accuracy")
    ax2.plot(epochs, val_accs, label="Val Accuracy")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    path = os.path.join(SCREENSHOTS_DIR, "loss_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[train] Saved {path}")
    return path


def save_confusion_matrix(model, val_loader, device):
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs.cpu() >= 0.5).int().squeeze()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["cat", "dog"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Validation Set)")
    path = os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[train] Saved {path}")
    return path


def main():
    device = get_device()
    print(f"[train] Device: {device}")

    train_loader, val_loader, class_to_idx = build_loaders()
    print(f"[train] class_to_idx: {class_to_idx}")
    print(f"[train] Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    mlflow.set_tracking_uri(f"file://{os.path.join(REPO_ROOT, 'mlruns')}")
    mlflow.set_experiment("cats-dogs-classification")

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    with mlflow.start_run(run_name="SimpleCNN_5ep") as run:
        mlflow.log_params({
            "architecture": "SimpleCNN",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "optimizer": "Adam",
            "loss_fn": "BCELoss",
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "image_size": 224,
            "augmentations": "RandomHorizontalFlip,RandomRotation(10),ColorJitter(0.2,0.2)",
        })

        for epoch in range(1, EPOCHS + 1):
            t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
            v_loss, v_acc = run_epoch(model, val_loader, criterion, optimizer, device, training=False)
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            train_accs.append(t_acc)
            val_accs.append(v_acc)
            print(f"[train] Epoch {epoch}/{EPOCHS}  train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")
            mlflow.log_metrics({
                "train_loss": t_loss, "train_accuracy": t_acc,
                "val_loss": v_loss, "val_accuracy": v_acc,
            }, step=epoch)

        curves_path = save_curves(train_losses, val_losses, train_accs, val_accs)
        cm_path = save_confusion_matrix(model, val_loader, device)
        mlflow.log_artifact(curves_path)
        mlflow.log_artifact(cm_path)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[train] Model saved → {MODEL_PATH}")

        sample_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            sample_output = model.cpu()(sample_input)
        signature = mlflow.models.infer_signature(sample_input.numpy(), sample_output.numpy())
        mlflow.pytorch.log_model(model.cpu(), "model", signature=signature, input_example=sample_input.numpy())
        mlflow.set_tag("best_model", "true")

        run_id = run.info.run_id

    print(f"[train] MLflow run ID: {run_id}")

    train_report = [
        f"Model        : SimpleCNN",
        f"Device       : {device}",
        f"Epochs       : {EPOCHS}",
        f"Batch size   : {BATCH_SIZE}",
        f"Learning rate: {LR}",
        f"Train samples: {len(train_loader.dataset):,}",
        f"Val samples  : {len(val_loader.dataset):,}",
        "",
        "Per-epoch metrics:",
    ]
    for i, (tl, ta, vl, va) in enumerate(zip(train_losses, val_losses, train_accs, val_accs), 1):
        train_report.append(f"  Epoch {i}: train_loss={tl:.4f}  train_acc={ta:.4f}  val_loss={vl:.4f}  val_acc={va:.4f}")
    train_report += [
        "",
        f"Final val accuracy : {val_accs[-1]:.4f}",
        f"Model artifact     : {MODEL_PATH}",
        f"Loss curves        : {curves_path}",
        f"Confusion matrix   : {cm_path}",
    ]
    write_report("step_4_train", train_report)

    mlflow_report = [
        f"Experiment   : cats-dogs-classification",
        f"Run name     : SimpleCNN_5ep",
        f"Run ID       : {run_id}",
        f"Tracking URI : {os.path.join(REPO_ROOT, 'mlruns')}",
        "",
        "Logged parameters:",
        f"  architecture={SimpleCNN.__name__}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}",
        f"  optimizer=Adam, loss_fn=BCELoss, image_size=224",
        "",
        "Final epoch metrics:",
        f"  train_loss={train_losses[-1]:.4f}  train_acc={train_accs[-1]:.4f}",
        f"  val_loss={val_losses[-1]:.4f}    val_acc={val_accs[-1]:.4f}",
        "",
        "Logged artifacts: loss_curves.png, confusion_matrix.png, model/",
    ]
    write_report("step_5_mlflow", mlflow_report)


if __name__ == "__main__":
    main()
