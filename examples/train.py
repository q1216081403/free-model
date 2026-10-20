"""Training loop skeleton for free-model.

This is intentionally a *skeleton*: it demonstrates the shape of a training
pipeline but does NOT train a meaningful model (the bundled model is random).

TODO:
  - Swap the dummy dataset for your real data loader.
  - Replace the NumPy MLP in ``free_model/model.py`` with your framework of
    choice (PyTorch, etc.) and implement a real backward pass + optimizer.
  - Save the trained weights with ``model.save(...)``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from free_model import FreeModel, FreeModelConfig  # noqa: E402
from free_model.utils import set_seed  # noqa: E402


def dummy_dataset(n: int = 1024, input_dim: int = 768, num_classes: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, input_dim)).astype(np.float32)
    y = rng.integers(0, num_classes, size=n)
    return X, y


def main() -> None:
    set_seed(42)
    cfg = FreeModelConfig()
    model = FreeModel(cfg)

    X, y = dummy_dataset()
    print(f"Dataset: X={X.shape}, y={y.shape}")
    print("TODO: implement forward/backward + optimizer to actually train.")
    print("Saving a skeleton checkpoint for demonstration purposes only.")

    out = Path("checkpoints/free_model")
    model.save(out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
