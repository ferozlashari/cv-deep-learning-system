"""
A CNN built from scratch in NumPy: forward AND backward passes implemented
by hand (no autograd). This exists to run/test/train a real, working CNN in
an environment without PyTorch installed, and to demonstrate the underlying
math (convolution gradients, max-pool routing, softmax cross-entropy) rather
than just framework API calls.

Architecture: Conv2D(1->8, 3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense -> Softmax
"""

from __future__ import annotations

import numpy as np


class Conv2D:
    """2D convolution, valid padding, stride 1. Weight shape: (out_ch, in_ch, kh, kw)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))  # He initialization
        self.W = rng.normal(0, scale, (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)
        self.kernel_size = kernel_size
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, C_in, H, W) -> out: (N, C_out, H-k+1, W-k+1)"""
        N, C_in, H, W = x.shape
        C_out, _, kh, kw = self.W.shape
        H_out, W_out = H - kh + 1, W - kw + 1

        out = np.zeros((N, C_out, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i:i + kh, j:j + kw]  # (N, C_in, kh, kw)
                # sum over C_in, kh, kw for each output channel
                out[:, :, i, j] = np.tensordot(patch, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b

        self._cache = x
        return out

    def backward(self, dout: np.ndarray, lr: float) -> np.ndarray:
        """dout: (N, C_out, H_out, W_out). Returns dx: (N, C_in, H, W). Updates W, b in place."""
        x = self._cache
        N, C_in, H, W = x.shape
        C_out, _, kh, kw = self.W.shape
        H_out, W_out = dout.shape[2], dout.shape[3]

        dW = np.zeros_like(self.W)
        db = dout.sum(axis=(0, 2, 3))
        dx = np.zeros_like(x)

        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i:i + kh, j:j + kw]  # (N, C_in, kh, kw)
                # dW[o] += sum_n dout[n,o,i,j] * patch[n]
                dW += np.tensordot(dout[:, :, i, j], patch, axes=([0], [0]))
                # dx patch += sum_o dout[n,o,i,j] * W[o]
                dx[:, :, i:i + kh, j:j + kw] += np.tensordot(dout[:, :, i, j], self.W, axes=([1], [0]))

        self.W -= lr * dW
        self.b -= lr * db
        return dx


class ReLU:
    def __init__(self):
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray, lr: float = 0.0) -> np.ndarray:
        return dout * (self._cache > 0)


class MaxPool2D:
    def __init__(self, size: int = 2):
        self.size = size
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        s = self.size
        H_out, W_out = H // s, W // s
        out = np.zeros((N, C, H_out, W_out))
        argmax = np.zeros((N, C, H_out, W_out, 2), dtype=int)

        for i in range(H_out):
            for j in range(W_out):
                window = x[:, :, i * s:i * s + s, j * s:j * s + s]  # (N, C, s, s)
                flat = window.reshape(N, C, -1)
                idx = flat.argmax(axis=-1)
                out[:, :, i, j] = flat.max(axis=-1)
                argmax[:, :, i, j, 0] = idx // s
                argmax[:, :, i, j, 1] = idx % s

        self._cache = (x.shape, argmax)
        return out

    def backward(self, dout: np.ndarray, lr: float = 0.0) -> np.ndarray:
        x_shape, argmax = self._cache
        N, C, H, W = x_shape
        s = self.size
        H_out, W_out = dout.shape[2], dout.shape[3]
        dx = np.zeros(x_shape)

        for i in range(H_out):
            for j in range(W_out):
                for n in range(N):
                    for c in range(C):
                        oi, oj = argmax[n, c, i, j]
                        dx[n, c, i * s + oi, j * s + oj] += dout[n, c, i, j]
        return dx


class Flatten:
    def __init__(self):
        self._shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray, lr: float = 0.0) -> np.ndarray:
        return dout.reshape(self._shape)


class Dense:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0, scale, (in_features, out_features))
        self.b = np.zeros(out_features)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray, lr: float) -> np.ndarray:
        x = self._cache
        N = x.shape[0]
        dW = x.T @ dout
        db = dout.sum(axis=0)
        dx = dout @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """logits: (N, num_classes), labels: (N,) int. Returns (loss, dlogits)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    N = logits.shape[0]
    log_likelihood = -np.log(probs[np.arange(N), labels] + 1e-12)
    loss = log_likelihood.mean()

    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1
    dlogits /= N
    return loss, dlogits


class SimpleCNN:
    """Conv(1->8,3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense -> logits"""

    def __init__(self, img_size: int = 20, num_classes: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.conv = Conv2D(1, 8, 3, rng)
        self.relu = ReLU()
        self.pool = MaxPool2D(2)
        self.flatten = Flatten()

        conv_out = img_size - 3 + 1  # valid conv
        pool_out = conv_out // 2
        flat_dim = 8 * pool_out * pool_out
        self.dense = Dense(flat_dim, num_classes, rng)

        self.layers = [self.conv, self.relu, self.pool, self.flatten, self.dense]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dlogits: np.ndarray, lr: float) -> None:
        dout = dlogits
        for layer in reversed(self.layers):
            dout = layer.backward(dout, lr)

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return logits.argmax(axis=1)

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        logits = self.forward(x)
        loss, dlogits = softmax_cross_entropy(logits, y)
        self.backward(dlogits, lr)
        return loss
