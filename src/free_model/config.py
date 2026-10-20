"""Configuration for the free-model.

Edit ``FreeModelConfig`` to add the hyperparameters your real model needs.
Keep it a plain dataclass so it can be serialized to / from JSON.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class FreeModelConfig:
    # --- input / output shape ------------------------------------------------
    input_dim: int = 768
    num_classes: int = 2

    # --- architecture (placeholder MLP) --------------------------------------
    hidden_dims: Optional[List[int]] = None
    dropout: float = 0.1

    # --- misc -----------------------------------------------------------------
    seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]

    # --- serialization --------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FreeModelConfig":
        # ignore unknown keys for forward-compatibility
        known = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**known)
