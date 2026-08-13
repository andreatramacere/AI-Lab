"""Neural-network layers for ToyTorch."""

import random

from .module import Module
from .parameter import Parameter


class Linear(Module):
    """Fully connected linear layer.

    Computes:
        y = W @ x + b

    Shapes:
        x:      (in_features,)
        W:      (out_features, in_features)
        b:      (out_features,)
        output: (out_features,)
    """

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Small random weights break symmetry between neurons.
        weights = [
            [
                random.uniform(-0.1, 0.1)
                for _ in range(in_features)
            ]
            for _ in range(out_features)
        ]

        biases = [0.0 for _ in range(out_features)]

        self.weight = Parameter(weights)
        self.bias = Parameter(biases)

    def forward(self, x):
        """Apply the linear transformation."""
        if x.shape != (self.in_features,):
            raise ValueError(
                f"Linear expected input shape ({self.in_features},), "
                f"received {x.shape}."
            )

        return self.weight @ x + self.bias


class ReLU(Module):
    """Module wrapper around the differentiable ReLU operation."""

    def forward(self, x):
        """Apply ReLU element-wise."""
        return x.relu()
