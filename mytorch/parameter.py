"""Learnable parameter type for ToyTorch."""

from .tensor import Tensor


class Parameter(Tensor):
    """A Tensor that is learnable by default.

    Parameter is semantically different from a regular Tensor:
    it represents model state that should receive gradients and be updated
    by an optimizer.
    """

    def __init__(self, data):
        super().__init__(data, requires_grad=True)

    def __repr__(self):
        return (
            f"Parameter(data={self.data}, "
            f"shape={self.shape}, "
            f"grad={self.grad})"
        )
