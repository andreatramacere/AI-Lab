"""Core Tensor implementation for ToyTorch.

This module defines the Tensor class and a few recursive helpers used to
represent nested numeric data, infer shapes, and accumulate gradients.
"""


def _infer_shape(data):
    """Infer the shape of nested Python numeric data.

    Supported inputs:
    - int / float -> scalar tensor, shape ()
    - list -> one or more tensor dimensions

    The nested list must be rectangular. Ragged structures raise ValueError.
    """
    if isinstance(data, (int, float)):
        return ()

    if not isinstance(data, list):
        raise TypeError(
            "ToyTorch supports only int, float, and nested Python lists."
        )

    if len(data) == 0:
        return (0,)

    first_shape = _infer_shape(data[0])

    for item in data[1:]:
        if _infer_shape(item) != first_shape:
            raise ValueError(
                "Ragged tensor: all nested elements must have the same shape."
            )

    return (len(data),) + first_shape


def _copy_nested(value):
    """Recursively copy a scalar or nested list of scalars."""
    if isinstance(value, (int, float)):
        return value

    return [_copy_nested(item) for item in value]


def _add_nested(a, b):
    """Recursively add two equally-shaped nested numeric structures."""
    if isinstance(a, (int, float)):
        return a + b

    return [_add_nested(x, y) for x, y in zip(a, b)]


class Tensor:
    """A minimal differentiable tensor used by ToyTorch.

    Attributes:
        data: Scalar or nested Python lists containing numeric values.
        shape: Tuple describing tensor dimensions.
        requires_grad: Whether gradients should be propagated to this tensor.
        grad: Accumulated gradient. None means no gradient has been computed yet.
        creator: Operation instance that produced this tensor, or None for leaves.
    """

    def __init__(self, data, requires_grad=False):
        """Create a Tensor from scalar or nested-list numeric data."""
        self.data = data
        self.shape = _infer_shape(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = None

    def __repr__(self):
        """Return a concise developer-friendly representation."""
        return (
            f"Tensor(data={self.data}, "
            f"shape={self.shape}, "
            f"requires_grad={self.requires_grad})"
        )

    def __add__(self, other):
        """Return element-wise addition of two tensors."""
        from .operations import Add
        return Add()(self, other)

    def __sub__(self, other):
        """Return element-wise subtraction of two tensors."""
        from .operations import Subtract
        return Subtract()(self, other)

    def __mul__(self, other):
        """Return element-wise multiplication of two tensors."""
        from .operations import Multiply
        return Multiply()(self, other)

    def __matmul__(self, other):
        """Return matrix-vector multiplication for the currently supported case."""
        from .operations import MatMul
        return MatMul()(self, other)

    def __pow__(self, exponent):
        """Raise every tensor element to a fixed numeric exponent."""
        from .operations import Power
        return Power(exponent)(self)

    def sum(self):
        """Reduce all tensor elements to a scalar sum."""
        from .operations import Sum
        return Sum()(self)

    def relu(self):
        """Apply ReLU element-wise."""
        from .operations import ReLU
        return ReLU()(self)

    def _accumulate_grad(self, grad):
        """Accumulate a new gradient contribution into this tensor.

        Gradients are accumulated rather than overwritten because the same tensor
        can influence the loss through multiple branches of the computational graph.
        """
        if self.grad is None:
            self.grad = _copy_nested(grad)
            return

        self.grad = _add_nested(self.grad, grad)

    def backward(self, grad=None):
        """Backpropagate gradients recursively through the computational graph.

        If `grad` is omitted, this method is valid only for scalar tensors.
        In that case, the seed gradient is 1 because dL/dL = 1.
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.shape != ():
                raise RuntimeError(
                    "backward() without an explicit grad is allowed only "
                    "for scalar tensors."
                )
            grad = 1.0

        self._accumulate_grad(grad)

        if self.creator is None:
            return

        input_grads = self.creator.backward(grad)

        for tensor, tensor_grad in zip(self.creator.inputs, input_grads):
            if tensor.requires_grad:
                tensor.backward(tensor_grad)
