"""Export a trained free-model checkpoint to a portable format.

Supported targets (implement as needed):
  - ONNX        : requires ``onnx`` + ``onnxruntime`` (or ``torch.onnx``)
  - safetensors : requires ``safetensors``

This script is currently a stub: it reloads the checkpoint and prints the layer
shapes. Fill in the exporter for your target format before publishing weights.
"""

import sys
from pathlib import Path

import numpy as np

# allow running without installing the package (package lives under src/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from free_model import FreeModel  # noqa: E402


def export_onnx(model: FreeModel, path: str) -> None:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        raise SystemExit("Install `onnx` + `onnxruntime` to export ONNX.")
    # TODO: build an ONNX graph from model.weights / model.biases
    raise NotImplementedError("ONNX export is not implemented yet.")


def main() -> None:
    ckpt = Path("checkpoints/free_model")
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt}; run examples/train.py first.")

    model = FreeModel.load(ckpt)
    print("Loaded model. Layers:")
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        print(f"  layer {i}: W{w.shape} b{b.shape}")

    print("TODO: call export_onnx(...) or a safetensors exporter to ship weights.")


if __name__ == "__main__":
    main()
