"""
Test suite for numpy_cnn.py and dataset.py.

Includes numerical gradient checking - the standard way to verify a hand-written
backward pass is mathematically correct (compares analytic gradients from
backward() against finite-difference approximations of the loss).

Run with: python3 tests/test_all.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.dataset import generate_dataset, generate_image, CLASSES
from src.numpy_cnn import SimpleCNN, Conv2D, Dense, MaxPool2D, softmax_cross_entropy

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


def numerical_gradient_check(model, x, y, param_getter, idx, eps=1e-4):
    """Central-difference gradient check for a single scalar parameter."""

    def loss_fn():
        logits = model.forward(x)
        loss, _ = softmax_cross_entropy(logits, y)
        return loss

    param = param_getter()
    orig = param[idx]
    param[idx] = orig + eps
    loss_plus = loss_fn()
    param[idx] = orig - eps
    loss_minus = loss_fn()
    param[idx] = orig
    return (loss_plus - loss_minus) / (2 * eps)


print("== dataset generation ==")
X, y = generate_dataset(n_per_class=20, seed=1)
check("correct shape (N, 1, H, W)", X.shape == (60, 1, 20, 20))
check("pixel values normalized to [0, 1]", X.min() >= 0 and X.max() <= 1)
check("balanced classes", all((y == i).sum() == 20 for i in range(3)))
check("labels within valid range", set(y.tolist()) == {0, 1, 2})

print("== layer shapes ==")
rng = np.random.default_rng(0)
conv = Conv2D(1, 8, 3, rng)
x = rng.normal(0, 1, (4, 1, 20, 20))
out = conv.forward(x)
check("conv output shape correct", out.shape == (4, 8, 18, 18))

pool = MaxPool2D(2)
pooled = pool.forward(out)
check("maxpool output shape correct", pooled.shape == (4, 8, 9, 9))

dense = Dense(8 * 9 * 9, 3, rng)
flat = pooled.reshape(4, -1)
logits = dense.forward(flat)
check("dense output shape correct", logits.shape == (4, 3))

print("== numerical gradient checking (verifies hand-written backprop is correct) ==")
model = SimpleCNN(img_size=10, num_classes=3, seed=1)
x_small = rng.normal(0, 0.1, (3, 1, 10, 10))
y_small = np.array([0, 1, 2])

# Run one full forward+backward to populate caches and get analytic gradients.
logits = model.forward(x_small)
loss, dlogits = softmax_cross_entropy(logits, y_small)
d = model.dense.backward(dlogits, lr=0.0)  # lr=0: computes gradient, applies no-op update
d = model.flatten.backward(d)
d = model.pool.backward(d)
d = model.relu.backward(d)
# Recompute conv dW analytically (mirrors Conv2D.backward internals) without mutating weights.
x_cache = model.conv._cache
kh, kw = model.conv.kernel_size, model.conv.kernel_size
H_out, W_out = d.shape[2], d.shape[3]
analytic_conv_dW = np.zeros_like(model.conv.W)
for i in range(H_out):
    for j in range(W_out):
        patch = x_cache[:, :, i:i + kh, j:j + kw]
        analytic_conv_dW += np.tensordot(d[:, :, i, j], patch, axes=([0], [0]))

for idx in [(0, 0, 0, 0), (3, 0, 1, 2), (7, 0, 2, 2)]:
    numerical = numerical_gradient_check(model, x_small, y_small, lambda: model.conv.W, idx)
    analytic = analytic_conv_dW[idx]
    rel_error = abs(numerical - analytic) / (abs(numerical) + abs(analytic) + 1e-8)
    check(f"conv weight gradient {idx} matches numerical (rel_error={rel_error:.2e})", rel_error < 1e-3)

dense_dW = model.dense._cache.T @ dlogits
for idx in [(0, 0), (5, 1), (10, 2)]:
    numerical = numerical_gradient_check(model, x_small, y_small, lambda: model.dense.W, idx)
    analytic = dense_dW[idx]
    rel_error = abs(numerical - analytic) / (abs(numerical) + abs(analytic) + 1e-8)
    check(f"dense weight gradient {idx} matches numerical (rel_error={rel_error:.2e})", rel_error < 1e-3)

print("== end-to-end training (should genuinely learn) ==")
X, y = generate_dataset(n_per_class=100, seed=42)
n_train = 240
X_train, y_train, X_test, y_test = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

model = SimpleCNN(img_size=20, num_classes=3, seed=7)
initial_logits = model.forward(X_test[:20])
initial_loss, _ = softmax_cross_entropy(initial_logits, y_test[:20])

for epoch in range(6):
    perm = np.random.default_rng(epoch).permutation(len(X_train))
    for i in range(0, len(X_train), 16):
        idx_batch = perm[i:i + 16]
        model.train_step(X_train[idx_batch], y_train[idx_batch], lr=0.05)

final_logits = model.forward(X_test[:20])
final_loss, _ = softmax_cross_entropy(final_logits, y_test[:20])
final_test_acc = (model.predict(X_test) == y_test).mean()

check("loss decreased substantially during training", final_loss < initial_loss * 0.5)
check("test accuracy well above random chance (1/3)", final_test_acc > 0.7)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
