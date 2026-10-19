"""
Trains the from-scratch NumPy CNN on the synthetic shape dataset and saves
a training curve plot + confusion matrix + metrics.json.

Run with: python3 train_numpy_cnn.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import CLASSES, generate_dataset, save_sample_images
from src.numpy_cnn import SimpleCNN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_training_curve(history: dict, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["loss"], marker="o")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(history["train_acc"], marker="o", label="Train")
    ax2.plot(history["test_acc"], marker="o", label="Test")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Confusion Matrix - NumPy CNN")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main(n_per_class: int = 150, epochs: int = 10, batch_size: int = 16, lr: float = 0.05, seed: int = 42) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)

    X, y = generate_dataset(n_per_class=n_per_class, seed=seed)
    save_sample_images(X, y, os.path.join(OUTPUT_DIR, "samples"), n=9)

    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    model = SimpleCNN(img_size=X.shape[-1], num_classes=len(CLASSES), seed=7)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        losses = []
        for i in range(0, len(X_train), batch_size):
            xb, yb = X_train[i:i + batch_size], y_train[i:i + batch_size]
            losses.append(model.train_step(xb, yb, lr))

        train_acc = float((model.predict(X_train) == y_train).mean())
        test_acc = float((model.predict(X_test) == y_test).mean())
        history["loss"].append(float(np.mean(losses)))
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch + 1}/{epochs}: loss={history['loss'][-1]:.4f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

    plot_training_curve(history, os.path.join(OUTPUT_DIR, "training_curve.png"))

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, len(CLASSES))
    plot_confusion_matrix(cm, CLASSES, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    results = {
        "final_train_accuracy": history["train_acc"][-1],
        "final_test_accuracy": history["test_acc"][-1],
        "final_loss": history["loss"][-1],
        "epochs": epochs,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
