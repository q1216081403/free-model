"""free_model — a free, open-source model project skeleton."""

from .config import FreeModelConfig
from .model import FreeModel
from . import utils

__all__ = ["FreeModel", "FreeModelConfig", "utils"]
__version__ = "0.1.0"
