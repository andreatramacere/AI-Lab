"""Small numerical smoke tests for ToyTorch."""

from .tensor import Tensor
from .layers import Linear
from .losses import MSELoss
from .module import Module
from .optim import SGD


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


def test_add_broadcast_forward_and_backward():
    """Check matrix-vector broadcasting and gradient reduction."""
    matrix = Tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        requires_grad=True,
    )
    bias = Tensor([10.0, 20.0, 30.0], requires_grad=True)

    output = matrix + bias
    output.sum().backward()

    assert output.shape == (2, 3)
    assert output.data == [
        [11.0, 22.0, 33.0],
        [14.0, 25.0, 36.0],
    ]
    assert matrix.grad == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert bias.grad == [2.0, 2.0, 2.0]


def test_multiply_broadcast_backward():
    """Check local derivatives followed by broadcast gradient reduction."""
    matrix = Tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        requires_grad=True,
    )
    scale = Tensor([2.0, 3.0, 4.0], requires_grad=True)

    (matrix * scale).sum().backward()

    assert matrix.grad == [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]
    assert scale.grad == [5.0, 7.0, 9.0]


def test_scalar_broadcast():
    """Check that a scalar broadcasts over every tensor dimension."""
    matrix = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    scalar = Tensor(2.0, requires_grad=True)

    output = matrix * scalar
    output.sum().backward()

    assert output.data == [[2.0, 4.0], [6.0, 8.0]]
    assert matrix.grad == [[2.0, 2.0], [2.0, 2.0]]
    assert scalar.grad == 10.0


def test_incompatible_broadcast_shapes():
    """Check that incompatible dimensions fail before graph construction."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([1.0, 2.0, 3.0])

    try:
        a + b
    except ValueError as error:
        assert "not broadcastable" in str(error)
    else:
        raise AssertionError("Expected incompatible shapes to raise ValueError")


def test_broadcast_singleton_and_missing_dimensions():
    """Check broadcasting across both singleton and missing dimensions."""
    a = Tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ]
    )
    b = Tensor(
        [
            [[10.0], [20.0], [30.0], [40.0]],
        ]
    )

    output = a + b

    assert output.shape == (2, 4, 3)
    assert output.data[0] == [
        [11.0, 12.0, 13.0],
        [21.0, 22.0, 23.0],
        [31.0, 32.0, 33.0],
        [41.0, 42.0, 43.0],
    ]


def test_empty_dimension_broadcast():
    """Check that a singleton dimension adopts an empty dimension."""
    empty = Tensor([])
    singleton = Tensor([1.0])

    output = empty + singleton

    assert output.shape == (0,)
    assert output.data == []


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


def test_backward_accumulates_parameter_gradients():
    """Check that repeated backward passes accumulate parameter gradients."""
    layer = Linear(1, 1)
    layer.weight.data = [[2.0]]
    layer.bias.data = [0.0]

    loss = layer(Tensor([3.0])).sum()
    loss.backward()
    loss.backward()

    assert layer.weight.grad == [[6.0]]
    assert layer.bias.grad == [2.0]


def test_sgd_zero_grad():
    """Check that zero_grad clears every optimizer-owned gradient."""
    layer = Linear(1, 1)
    optimizer = SGD(layer.parameters(), lr=0.1)

    layer(Tensor([2.0])).sum().backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None

    optimizer.zero_grad()

    assert layer.weight.grad is None
    assert layer.bias.grad is None


def test_module_discovers_nested_parameters():
    """Check recursive parameter discovery through the module hierarchy."""
    class TwoLayerNet(Module):
        def __init__(self):
            self.layer1 = Linear(1, 2)
            self.layer2 = Linear(2, 1)

        def forward(self, x):
            return self.layer2(self.layer1(x))

    model = TwoLayerNet()

    assert model.parameters() == [
        model.layer1.weight,
        model.layer1.bias,
        model.layer2.weight,
        model.layer2.bias,
    ]


def test_sgd_step_updates_parameters_only():
    """Check that SGD updates parameters without mutating input data."""
    layer = Linear(1, 1)
    layer.weight.data = [[1.0]]
    layer.bias.data = [0.0]
    optimizer = SGD(layer.parameters(), lr=0.1)
    x = Tensor([2.0], requires_grad=True)

    layer(x).sum().backward()
    optimizer.step()

    assert layer.weight.data == [[0.8]]
    assert layer.bias.data == [-0.1]
    assert x.data == [2.0]


def test_single_training_step_reduces_loss():
    """Check a complete deterministic forward-backward-update cycle."""
    layer = Linear(1, 1)
    layer.weight.data = [[1.0]]
    layer.bias.data = [0.0]
    loss_fn = MSELoss()
    optimizer = SGD(layer.parameters(), lr=0.1)
    x = Tensor([2.0])
    target = Tensor([5.0])

    loss_before = loss_fn(layer(x), target)
    loss_before.backward()
    optimizer.step()
    loss_after = loss_fn(layer(x), target)

    assert_close(loss_before.data, 9.0)
    assert_close(layer.weight.data[0][0], 2.2)
    assert_close(layer.bias.data[0], 0.6)
    assert loss_after.data < loss_before.data
    assert_close(loss_after.data, 0.0)


def run_all():
    """Run every ToyTorch smoke test."""
    tests = [
        test_add_backward,
        test_multiply_backward,
        test_add_broadcast_forward_and_backward,
        test_multiply_broadcast_backward,
        test_scalar_broadcast,
        test_incompatible_broadcast_shapes,
        test_broadcast_singleton_and_missing_dimensions,
        test_empty_dimension_broadcast,
        test_matmul_backward,
        test_relu_backward,
        test_mse_backward,
        test_linear_backward,
        test_backward_accumulates_parameter_gradients,
        test_sgd_zero_grad,
        test_module_discovers_nested_parameters,
        test_sgd_step_updates_parameters_only,
        test_single_training_step_reduces_loss,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    run_all()
