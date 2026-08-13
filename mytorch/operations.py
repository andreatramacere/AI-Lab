"""Differentiable primitive operations for ToyTorch.

Each Operation:
1. stores its input tensors,
2. runs a local forward computation,
3. creates an output Tensor connected to the graph,
4. implements the local backward rule used by autograd.
"""

from tensor import Tensor


def _is_number(value):
    """Return True for supported scalar numeric values."""
    return isinstance(value, (int, float))


def _map_unary(data, fn):
    """Apply a unary function recursively to nested numeric data."""
    if _is_number(data):
        return fn(data)

    return [_map_unary(item, fn) for item in data]


def _map_binary(a, b, fn):
    """Apply a binary function recursively to equally-shaped nested data."""
    if _is_number(a):
        return fn(a, b)

    return [_map_binary(x, y, fn) for x, y in zip(a, b)]


def _flatten_sum(data):
    """Recursively sum all scalar values in nested numeric data."""
    if _is_number(data):
        return data

    return sum(_flatten_sum(item) for item in data)


def _fill_like(data, value):
    """Create a nested structure shaped like data and filled with value."""
    if _is_number(data):
        return value

    return [_fill_like(item, value) for item in data]


class Operation:
    """Base class for differentiable ToyTorch operations."""

    def __call__(self, *inputs):
        """Execute forward logic and connect the result to the computation graph."""
        self.inputs = inputs
        data = self.forward(*inputs)

        requires_grad = any(tensor.requires_grad for tensor in inputs)

        result = Tensor(data, requires_grad=requires_grad)
        result.creator = self
        self.output = result

        return result

    def forward(self, *inputs):
        """Compute raw output data from input tensors."""
        raise NotImplementedError

    def backward(self, grad_output):
        """Return one gradient per input tensor."""
        raise NotImplementedError

    @staticmethod
    def validate_same_shape(a, b):
        """Require identical shapes for element-wise binary operations."""
        if a.shape != b.shape:
            raise ValueError(
                f"Incompatible shapes: {a.shape} and {b.shape}."
            )


class Add(Operation):
    """Element-wise addition."""

    def forward(self, a, b):
        self.validate_same_shape(a, b)
        return _map_binary(a.data, b.data, lambda x, y: x + y)

    def backward(self, grad_output):
        return grad_output, grad_output


class Subtract(Operation):
    """Element-wise subtraction."""

    def forward(self, a, b):
        self.validate_same_shape(a, b)
        return _map_binary(a.data, b.data, lambda x, y: x - y)

    def backward(self, grad_output):
        grad_a = grad_output
        grad_b = _map_unary(grad_output, lambda g: -g)
        return grad_a, grad_b


class Multiply(Operation):
    """Element-wise multiplication."""

    def forward(self, a, b):
        self.validate_same_shape(a, b)
        return _map_binary(a.data, b.data, lambda x, y: x * y)

    def backward(self, grad_output):
        a, b = self.inputs

        grad_a = _map_binary(
            grad_output,
            b.data,
            lambda g, y: g * y,
        )
        grad_b = _map_binary(
            grad_output,
            a.data,
            lambda g, x: g * x,
        )

        return grad_a, grad_b


class Power(Operation):
    """Element-wise fixed-exponent power operation."""

    def __init__(self, exponent):
        self.exponent = exponent

    def forward(self, x):
        return _map_unary(
            x.data,
            lambda value: value ** self.exponent,
        )

    def backward(self, grad_output):
        x, = self.inputs
        n = self.exponent

        local_derivative = _map_unary(
            x.data,
            lambda value: n * (value ** (n - 1)),
        )

        grad_x = _map_binary(
            grad_output,
            local_derivative,
            lambda g, local: g * local,
        )

        return (grad_x,)


class Sum(Operation):
    """Reduce all tensor elements to a scalar sum."""

    def forward(self, x):
        return _flatten_sum(x.data)

    def backward(self, grad_output):
        x, = self.inputs
        grad_x = _fill_like(x.data, grad_output)
        return (grad_x,)


class ReLU(Operation):
    """Rectified Linear Unit applied element-wise."""

    def forward(self, x):
        return _map_unary(
            x.data,
            lambda value: max(0.0, value),
        )

    def backward(self, grad_output):
        x, = self.inputs

        mask = _map_unary(
            x.data,
            lambda value: 1.0 if value > 0 else 0.0,
        )

        grad_x = _map_binary(
            grad_output,
            mask,
            lambda g, active: g * active,
        )

        return (grad_x,)


class MatMul(Operation):
    """Matrix-vector multiplication.

    Currently supported:
        matrix.shape == (rows, cols)
        vector.shape == (cols,)

    Output:
        shape == (rows,)
    """

    def forward(self, matrix, vector):
        if len(matrix.shape) != 2:
            raise ValueError(
                "The first operand of MatMul must be a 2D matrix."
            )

        if len(vector.shape) != 1:
            raise ValueError(
                "The second operand of MatMul must be a 1D vector."
            )

        rows, cols = matrix.shape

        if vector.shape[0] != cols:
            raise ValueError(
                f"Incompatible MatMul shapes: "
                f"{matrix.shape} @ {vector.shape}."
            )

        result = []

        for row in matrix.data:
            total = sum(
                weight * value
                for weight, value in zip(row, vector.data)
            )
            result.append(total)

        return result

    def backward(self, grad_output):
        matrix, vector = self.inputs
        rows, cols = matrix.shape

        grad_matrix = []
        for i in range(rows):
            row_grad = []
            for j in range(cols):
                row_grad.append(
                    grad_output[i] * vector.data[j]
                )
            grad_matrix.append(row_grad)

        grad_vector = []
        for j in range(cols):
            total = 0.0
            for i in range(rows):
                total += grad_output[i] * matrix.data[i][j]
            grad_vector.append(total)

        return grad_matrix, grad_vector
