"""
Trains src.torch_model.ShapeCNN on the synthetic shape dataset using PyTorch.
Requires `torch` (not installed in this sandbox - install per README, then run):

    python3 train_torch.py

Reuses src/dataset.py so the PyTorch model trains on the exact same data as
the verified NumPy CNN (train_numpy_cnn.py), making their results directly
comparable.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import CLASSES, generate_dataset
from src.torch_model import ShapeCNN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main(n_per_class: int = 300, epochs: int = 15, batch_size: int = 32, lr: float = 1e-3, seed: int = 42) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    X, y = generate_dataset(n_per_class=n_per_class, seed=seed)
    n_train = int(len(X) * 0.8)

    X_train = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y[:n_train], dtype=torch.long)
    X_test = torch.tensor(X[n_train:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y[n_train:], dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model = ShapeCNN(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        history["loss"].append(sum(epoch_losses) / len(epoch_losses))
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch + 1}/{epochs}: loss={history['loss'][-1]:.4f} test_acc={test_acc:.3f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "torch_model.pt"))
    with open(os.path.join(OUTPUT_DIR, "torch_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    main()
