# Deep Learning Computer Vision System

A CNN image classifier (circle / square / triangle) implemented **two ways**:

1. **`src/numpy_cnn.py`** — a CNN built from scratch in raw NumPy: convolution, ReLU, max-pooling, and dense layers, each with a hand-written forward *and backward* pass (no autograd). Trained and verified in this repo.
2. **`src/torch_model.py`** — the production version: the same architecture (deeper, with BatchNorm) built on PyTorch, for real-world training speed and GPU support.

## Why two implementations?

Anyone can call `nn.Conv2d`. Writing the backward pass by hand — and proving it's correct with **numerical gradient checking** — demonstrates understanding of what's actually happening inside a CNN during training, not just the framework API. Both are here: the from-scratch version as proof, the PyTorch version as the deployable production model.

## Verified results (from-scratch NumPy CNN)

Trained on 450 synthetic 20×20 grayscale images (150/class: circle, square, triangle — randomized position/size + pixel noise):

```
epoch 1/10:  loss=0.7914  train_acc=0.867  test_acc=0.800
epoch 5/10:  loss=0.2504  train_acc=0.936  test_acc=0.878
epoch 10/10: loss=0.1204  train_acc=0.978  test_acc=0.933
```

**Gradient correctness**: every layer's hand-written `backward()` was checked against finite-difference numerical gradients (the standard method for verifying backprop implementations). All checks pass with relative error < 1e-8 — see `tests/test_all.py`.

## Project structure

```
src/
├── dataset.py           Synthetic shape image generator (PIL)
├── numpy_cnn.py           From-scratch CNN: Conv2D, ReLU, MaxPool2D, Dense, softmax+CE (with backward())
└── torch_model.py          Production PyTorch CNN (Conv-BN-ReLU-Pool x2 + FC)
train_numpy_cnn.py          Trains & evaluates the NumPy CNN, saves plots (runs anywhere)
train_torch.py               Trains the PyTorch CNN (requires torch)
tests/
└── test_all.py               15 tests: shapes, numerical gradient checks, end-to-end training
outputs/
├── training_curve.png, confusion_matrix.png, metrics.json
└── samples/                  9 example generated images
```

## Setup & Run

### From-scratch NumPy CNN (runs with just numpy + matplotlib + pillow)

```bash
pip install -r requirements.txt   # or just: pip install numpy matplotlib pillow
python3 train_numpy_cnn.py
python3 tests/test_all.py
```

### Production PyTorch CNN

```bash
pip install torch torchvision
python3 train_torch.py
```

Both scripts train on the exact same `src/dataset.py` data, so their results are directly comparable — the PyTorch version should reach similar or better accuracy, faster, thanks to Adam + BatchNorm + a deeper architecture.

## The bug numerical gradient checking caught

The first version of `Conv2D.backward` and `Dense.backward` divided the weight gradient by the batch size `N` — but `softmax_cross_entropy`'s `dlogits` *already* included a `1/N` factor. That's a double-division bug: gradients came out `N` times too small, which would have made training silently crawl instead of erroring out (the most dangerous kind of ML bug — it "works," just badly).

Numerical gradient checking caught it immediately: for a batch of 2, the analytic gradient was exactly half the finite-difference gradient. Fixed by removing the redundant `/N` from both layers' weight/bias updates. Documented here because catching this kind of silent-but-wrong bug is exactly what this technique is for.

## Known limitations / next steps

- The NumPy conv layer uses plain nested loops (no im2col), so it's fine for small images/datasets but not competitive with PyTorch's cuDNN kernels at scale — that's expected; it's a correctness reference, not a performance target.
- Synthetic shapes rather than a real-world dataset (e.g. CIFAR-10) — swap `src/dataset.py` for a `torchvision.datasets` loader to point `train_torch.py` at real images.
- No GPU used/tested here; `train_torch.py` auto-detects CUDA if available.

## Tech stack

Python · PyTorch · NumPy · Matplotlib · Pillow
