"""
Generates a synthetic image classification dataset: grayscale 20x20 images of
circles, squares, and triangles, with randomized position, size, rotation,
and pixel noise - enough variation that a CNN has genuine work to do.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 20
CLASSES = ["circle", "square", "triangle"]


def _draw_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)


def _draw_square(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=255)


def _draw_triangle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
    draw.polygon(points, fill=255)


_DRAW_FUNCS = {"circle": _draw_circle, "square": _draw_square, "triangle": _draw_triangle}


def generate_image(label: str, rng: np.random.Generator, noise_std: float = 12.0) -> np.ndarray:
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    draw = ImageDraw.Draw(img)

    r = int(rng.integers(5, 9))
    cx = int(rng.integers(r + 1, IMG_SIZE - r - 1))
    cy = int(rng.integers(r + 1, IMG_SIZE - r - 1))
    _DRAW_FUNCS[label](draw, cx, cy, r)

    arr = np.array(img).astype(np.float32)
    noise = rng.normal(0, noise_std, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return arr / 255.0  # normalize to [0, 1]


def generate_dataset(n_per_class: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for class_idx, label in enumerate(CLASSES):
        for _ in range(n_per_class):
            images.append(generate_image(label, rng))
            labels.append(class_idx)
    X = np.stack(images)[:, None, :, :]  # (N, 1, H, W) - channel-first, like PyTorch expects
    y = np.array(labels, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def save_sample_images(X: np.ndarray, y: np.ndarray, out_dir: str, n: int = 9) -> list[str]:
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(min(n, len(X))):
        img = Image.fromarray((X[i, 0] * 255).astype(np.uint8))
        img = img.resize((100, 100), Image.NEAREST)
        path = os.path.join(out_dir, f"sample_{i}_{CLASSES[y[i]]}.png")
        img.save(path)
        paths.append(path)
    return paths
