"""Inference example for free-model.

Runs a forward pass through the model. If a checkpoint exists it is loaded;
otherwise a fresh (random) model is used so the script always runs.
"""

import sys
from pathlib import Path

import numpy as np

# allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from free_model import FreeModel, FreeModelConfig  # noqa: E402


def main() -> None:
    ckpt = Path("checkpoints/free_model")
    if ckpt.exists():
        model = FreeModel.load(ckpt)
        print(f"Loaded model from {ckpt}")
    else:
        model = FreeModel(FreeModelConfig())
        print("No checkpoint found — using a fresh random model.")

    cfg = model.config
    x = np.random.randn(3, cfg.input_dim).astype(np.float32)

    logits = model.forward(x)
    probs = model.predict_proba(x)
    preds = model.predict(x)

    print(f"input shape : {x.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"predictions : {preds.tolist()}")
    print(f"max proba   : {probs.max(axis=-1).round(4).tolist()}")


if __name__ == "__main__":
    main()
