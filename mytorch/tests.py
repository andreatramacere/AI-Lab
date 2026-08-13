"""Small numerical smoke tests for ToyTorch."""

from tensor import Tensor
from layers import Linear
from losses import MSELoss


def assert_close(actual, expected, tol=1e-9):
    """Assert that two scalar values are numerically close."""
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"Expected {expected}, received {actual}"
        )


def test_add_backward():
    """Check gradient accumulation for x + x."""
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = x + x
    loss = y.sum()
    loss.backward()

    assert x.grad == [2.0, 2.0]


def test_multiply_backward():
    """Check local multiplication gradients."""
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)

    loss = (a * b).sum()
    loss.backward()

    assert a.grad == [3.0, 4.0]
    assert b.grad == [1.0, 2.0]


def test_matmul_backward():
    """Check matrix-vector forward and backward."""
    matrix = Tensor(
        [
            [0.5, 1.0, -2.0],
            [1.5, -1.0, 0.2],
        ],
        requires_grad=True,
    )
    vector = Tensor(
        [2.0, 4.0, 1.0],
        requires_grad=True,
    )

    output = matrix @ vector
    loss = output.sum()
    loss.backward()

    assert output.data == [3.0, -0.8]
    assert matrix.grad == [
        [2.0, 4.0, 1.0],
        [2.0, 4.0, 1.0],
    ]
    assert vector.grad == [2.0, 0.0, -1.8]


def test_relu_backward():
    """Check that ReLU blocks gradients for non-positive inputs."""
    x = Tensor(
        [-3.0, -1.0, 2.0, 5.0],
        requires_grad=True,
    )

    loss = x.relu().sum()
    loss.backward()

    assert x.grad == [0.0, 0.0, 1.0, 1.0]


def test_mse_backward():
    """Check MSE loss value and prediction gradient."""
    prediction = Tensor(
        [2.0, 4.0],
        requires_grad=True,
    )
    target = Tensor([3.0, 7.0])

    loss = MSELoss()(prediction, target)
    loss.backward()

    assert_close(loss.data, 5.0)
    assert prediction.grad == [-1.0, -3.0]


def test_linear_backward():
    """Check a deterministic Linear layer gradient."""
    layer = Linear(3, 2)

    layer.weight.data = [
        [0.5, 1.0, -2.0],
        [1.5, -1.0, 0.2],
    ]
    layer.bias.data = [1.0, 2.0]

    x = Tensor([2.0, 4.0, 1.0])
    loss = layer(x).sum()
    loss.backward()

    assert layer.weight.grad == [
        [2.0, 4.0, 1.0],
        [2.0, 4.0, 1.0],
    ]
    assert layer.bias.grad == [1.0, 1.0]


def run_all():
    """Run every ToyTorch smoke test."""
    tests = [
        test_add_backward,
        test_multiply_backward,
        test_matmul_backward,
        test_relu_backward,
        test_mse_backward,
        test_linear_backward,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    run_all()
