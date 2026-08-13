"""Optimizers for ToyTorch."""


def _sgd_update(data, grad, lr):
    """Recursively apply one SGD update to nested parameter data."""
    if isinstance(data, (int, float)):
        return data - lr * grad

    return [
        _sgd_update(d, g, lr)
        for d, g in zip(data, grad)
    ]


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, parameters, lr=0.01):
        """Store learnable parameters and learning rate."""
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        """Update every parameter using its accumulated gradient."""
        for parameter in self.parameters:
            if parameter.grad is None:
                continue

            parameter.data = _sgd_update(
                parameter.data,
                parameter.grad,
                self.lr,
            )

    def zero_grad(self):
        """Clear previously accumulated gradients."""
        for parameter in self.parameters:
            parameter.grad = None
