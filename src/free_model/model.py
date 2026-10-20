"""The model.

This is a *placeholder* multi-layer perceptron implemented with NumPy so the
project runs with zero heavy dependencies. It is intentionally simple.

TODO: replace ``FreeModel`` with your real architecture (PyTorch / JAX / ONNX
runtime / …). Preserve the public surface ``forward``, ``predict``,
``predict_proba``, ``save`` and ``load`` so the examples and tests keep working.
"""

import json
from pathlib import Path
from typing import List

import numpy as np

from .config import FreeModelConfig


class FreeModel:
    def __init__(self, config: FreeModelConfig = None) -> None:
        self.config = config or FreeModelConfig()
        self.training = False
        rng = np.random.default_rng(self.config.seed)

        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.num_classes]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(dims) - 1):
            # He-ish initialization
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(rng.normal(0.0, scale, (dims[i], dims[i + 1])).astype(np.float32))
            self.biases.append(np.zeros(dims[i + 1], dtype=np.float32))

    # --- internals ------------------------------------------------------------
    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    # --- forward / predict ----------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = self._relu(x)
                if self.training and self.config.dropout > 0:
                    mask = (
                        np.random.rand(*x.shape) > self.config.dropout
                    ).astype(np.float32)
                    x = x * mask / (1.0 - self.config.dropout)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        logits = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=-1, keepdims=True)

    # --- persistence ----------------------------------------------------------
    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "weights.npz",
            **{f"w{i}": w for i, w in enumerate(self.weights)},
            **{f"b{i}": b for i, b in enumerate(self.biases)},
        )
        (path / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path) -> "FreeModel":
        path = Path(path)
        cfg = FreeModelConfig.from_dict(
            json.loads((path / "config.json").read_text(encoding="utf-8"))
        )
        obj = cls(cfg)
        data = np.load(path / "weights.npz")
        n_layers = len(cfg.hidden_dims) + 1
        obj.weights = [data[f"w{i}"] for i in range(n_layers)]
        obj.biases = [data[f"b{i}"] for i in range(n_layers)]
        return obj
