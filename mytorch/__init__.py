"""Public API for the educational MyTorch framework."""

from .layers import Linear, ReLU
from .losses import MSELoss
from .module import Module
from .optim import SGD
from .parameter import Parameter
from .tensor import Tensor

__all__ = [
    "Linear",
    "MSELoss",
    "Module",
    "Parameter",
    "ReLU",
    "SGD",
    "Tensor",
]

__version__ = "0.1.0"
