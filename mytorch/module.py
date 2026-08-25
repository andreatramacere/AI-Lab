"""Base Module abstraction for ToyTorch."""

from .parameter import Parameter


class Module:
    """Base class for model components.

    Subclasses implement forward(). A Module can contain Parameters and
    nested Modules. named_parameters() recursively preserves ownership paths,
    while parameters() exposes only the learnable values.
    """

    def __call__(self, *inputs):
        """Forward calls through the module using function-like syntax."""
        return self.forward(*inputs)

    def forward(self, *inputs):
        """Define the forward computation in subclasses."""
        raise NotImplementedError

    def named_parameters(self, prefix=""):
        """Return ``(path, Parameter)`` pairs from the module hierarchy."""
        named_params = []

        for name, value in self.__dict__.items():
            path = f"{prefix}.{name}" if prefix else name

            if isinstance(value, Parameter):
                named_params.append((path, value))
            elif isinstance(value, Module):
                named_params.extend(value.named_parameters(path))

        return named_params

    def parameters(self):
        """Return all Parameters owned directly or recursively by this module."""
        return [
            parameter
            for _, parameter in self.named_parameters()
        ]
