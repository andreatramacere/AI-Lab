"""Base Module abstraction for ToyTorch."""

from .parameter import Parameter


class Module:
    """Base class for model components.

    Subclasses implement forward(). A Module can contain Parameters and
    nested Modules. parameters() recursively discovers learnable parameters.
    """

    def __call__(self, *inputs):
        """Forward calls through the module using function-like syntax."""
        return self.forward(*inputs)

    def forward(self, *inputs):
        """Define the forward computation in subclasses."""
        raise NotImplementedError

    def parameters(self):
        """Return all Parameters owned directly or recursively by this module."""
        params = []

        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())

        return params
