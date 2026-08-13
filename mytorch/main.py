"""End-to-end ToyTorch demo.

This example trains a tiny multilayer network to approximate y = x^2
on a handful of scalar samples represented as one-element vectors.
"""

import random

from .tensor import Tensor
from .module import Module
from .layers import Linear, ReLU
from .losses import MSELoss
from .optim import SGD


class TinyNet(Module):
    """A minimal multilayer neural network.

    Architecture:
        Linear(1, 4)
        ReLU
        Linear(4, 1)
    """

    def __init__(self):
        self.layer1 = Linear(1, 4)
        self.relu = ReLU()
        self.layer2 = Linear(4, 1)

    def forward(self, x):
        """Compute one forward pass through the network."""
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def main():
    """Train TinyNet on a tiny y=x^2 dataset and print predictions."""
    random.seed(0)

    training_data = [
        (-2.0, 4.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 4.0),
    ]

    model = TinyNet()
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(300):
        total_loss = 0.0

        for x_value, target_value in training_data:
            x = Tensor([x_value])
            target = Tensor([target_value])

            optimizer.zero_grad()

            prediction = model(x)
            loss = loss_fn(prediction, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.data

        if epoch % 50 == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={total_loss:.6f}"
            )

    print("\nPredictions:")
    for x_value, target_value in training_data:
        prediction = model(Tensor([x_value]))
        print(
            f"x={x_value:>4.1f} "
            f"target={target_value:>4.1f} "
            f"prediction={prediction.data[0]:.4f}"
        )


if __name__ == "__main__":
    main()
