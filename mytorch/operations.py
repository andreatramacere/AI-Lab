"""Differentiable primitive operations for ToyTorch.

Each Operation:
1. stores its input tensors,
2. runs a local forward computation,
3. creates an output Tensor connected to the graph,
4. implements the local backward rule used by autograd.
"""

from .tensor import Tensor


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


def _broadcast_shape(shape_a, shape_b):
    """Return the common broadcast shape using right-aligned dimensions."""
    result = []
    rank = max(len(shape_a), len(shape_b))

    padded_a = (1,) * (rank - len(shape_a)) + shape_a
    padded_b = (1,) * (rank - len(shape_b)) + shape_b

    for dim_a, dim_b in zip(padded_a, padded_b):
        if dim_a == dim_b:
            result.append(dim_a)
            continue

        if dim_a == 1:
            result.append(dim_b)
            continue

        if dim_b == 1:
            result.append(dim_a)
            continue

        raise ValueError(
            f"Shapes {shape_a} and {shape_b} are not broadcastable."
        )

    return tuple(result)


def _build_nested(shape, value_at, prefix=()):
    """Build nested data with shape, obtaining scalar values by index."""
    if not shape:
        return value_at(prefix)

    return [
        _build_nested(shape[1:], value_at, prefix + (index,))
        for index in range(shape[0])
    ]


def _get_nested(data, index):
    """Read a scalar from nested data using a tuple index."""
    value = data
    for position in index:
        value = value[position]
    return value


def _project_index(output_index, input_shape):
    """Map an output index to an input index under broadcasting."""
    offset = len(output_index) - len(input_shape)
    return tuple(
        0 if dimension == 1 else output_index[offset + axis]
        for axis, dimension in enumerate(input_shape)
    )


def _broadcast_binary(a, shape_a, b, shape_b, fn):
    """Apply a binary function after broadcasting two nested structures."""
    output_shape = _broadcast_shape(shape_a, shape_b)

    def value_at(output_index):
        value_a = _get_nested(a, _project_index(output_index, shape_a))
        value_b = _get_nested(b, _project_index(output_index, shape_b))
        return fn(value_a, value_b)

    return _build_nested(output_shape, value_at)


def _iter_indices(shape, prefix=()):
    """Yield every scalar index in a shape."""
    if not shape:
        yield prefix
        return

    for index in range(shape[0]):
        yield from _iter_indices(shape[1:], prefix + (index,))


def _reduce_broadcast_gradient(grad, grad_shape, target_shape):
    """Sum a broadcasted gradient back to an operand's original shape."""
    if grad_shape == target_shape:
        return grad

    totals = {}

    for grad_index in _iter_indices(grad_shape):
        target_index = _project_index(grad_index, target_shape)
        totals[target_index] = (
            totals.get(target_index, 0.0)
            + _get_nested(grad, grad_index)
        )

    return _build_nested(
        target_shape,
        lambda index: totals.get(index, 0.0),
    )


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

class Add(Operation):
    """Element-wise addition."""

    def forward(self, a, b):
        return _broadcast_binary(
            a.data, a.shape,
            b.data, b.shape,
            lambda x, y: x + y,
        )

    def backward(self, grad_output):
        a, b = self.inputs
        output_shape = self.output.shape
        return (
            _reduce_broadcast_gradient(
                grad_output, output_shape, a.shape
            ),
            _reduce_broadcast_gradient(
                grad_output, output_shape, b.shape
            ),
        )


class Subtract(Operation):
    """Element-wise subtraction."""

    def forward(self, a, b):
        return _broadcast_binary(
            a.data, a.shape,
            b.data, b.shape,
            lambda x, y: x - y,
        )

    def backward(self, grad_output):
        a, b = self.inputs
        output_shape = self.output.shape
        grad_a = _reduce_broadcast_gradient(
            grad_output, output_shape, a.shape
        )
        grad_b = _reduce_broadcast_gradient(
            _map_unary(grad_output, lambda g: -g),
            output_shape,
            b.shape,
        )
        return grad_a, grad_b


class Multiply(Operation):
    """Element-wise multiplication."""

    def forward(self, a, b):
        return _broadcast_binary(
            a.data, a.shape,
            b.data, b.shape,
            lambda x, y: x * y,
        )

    def backward(self, grad_output):
        a, b = self.inputs

        output_shape = self.output.shape

        grad_a_full = _broadcast_binary(
            grad_output, output_shape,
            b.data, b.shape,
            lambda g, y: g * y,
        )
        grad_b_full = _broadcast_binary(
            grad_output, output_shape,
            a.data, a.shape,
            lambda g, x: g * x,
        )

        grad_a = _reduce_broadcast_gradient(
            grad_a_full, output_shape, a.shape
        )
        grad_b = _reduce_broadcast_gradient(
            grad_b_full, output_shape, b.shape
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
