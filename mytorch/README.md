# MyTorch

MyTorch is a small educational deep-learning framework built from scratch in
plain Python.

It currently includes:

- scalar, vector, and nested-list Tensor data
- a dynamic computational graph
- recursive reverse-mode autodiff
- gradient accumulation
- NumPy-style broadcasting for element-wise binary operations
- element-wise Add, Subtract, Multiply, Power, and ReLU
- Sum reduction
- matrix-vector MatMul
- learnable Parameter objects
- Module composition
- Linear and ReLU layers
- MSELoss
- SGD
- a tiny multilayer training example

## Install

```bash
python -m pip install -e .
```

The editable install exposes the public API:

```python
from mytorch import Linear, MSELoss, SGD, Tensor
```

MyTorch has no runtime dependencies outside the Python standard library.

## Run the tests

From the repository root:

```bash
python -m mytorch.tests
```

## Run the demo

```bash
mytorch-demo
```

or, without using the installed command:

```bash
python -m mytorch.main
```

## Current limitations

This is intentionally a teaching framework, not a production numerical library.

Not yet supported:

- general matrix-matrix multiplication
- batches
- NumPy-backed storage
- GPU execution
- dtype/device management
- topological backward scheduling
- graph detaching / no-grad mode
- advanced optimizers
- robust initialization schemes
