"""
Production CNN using PyTorch. Requires `torch` + `torchvision` (not installed
in this sandbox - see README). Architecturally mirrors src/numpy_cnn.py
(same conv->relu->pool->dense structure) but uses cuDNN-backed layers,
autograd, and batch normalization for real-world training speed and stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ShapeCNN(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 20x20 -> 10x10

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
