"""Loss functions for ToyTorch."""

from .tensor import Tensor


class MSELoss:
    """Mean Squared Error loss.

    Computes:
        mean((prediction - target) ** 2)

    The loss is built by composing differentiable ToyTorch operations, so it
    does not need a custom backward implementation.
    """

    def __call__(self, prediction, target):
        """Return a scalar MSE loss tensor."""
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have the same shape."
            )

        error = prediction - target
        squared_error = error ** 2

        if prediction.shape == ():
            return squared_error

        total = squared_error.sum()

        n = 1
        for dimension in prediction.shape:
            n *= dimension

        if n == 0:
            raise ValueError("MSELoss is undefined for an empty tensor.")

        scale = Tensor(1.0 / n)

        return total * scale
