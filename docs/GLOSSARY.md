# Glossary

## Tensor
A value-carrying object that combines numerical data with the context required for differentiable computation, including shape, gradient state, and the operation that created it.

## Operation
A transformation that consumes one or more Tensors, computes new values, and produces an output Tensor while preserving the information required for backward propagation.

## Computational Graph
The distributed graph that emerges from the links between Tensors and the Operations that produced them. It records the history of the forward computation.

## Autograd
The mechanism that traverses the computational graph backwards and composes local derivatives to compute gradients.

## Parameter
A Tensor with learnable semantics: it identifies a value that belongs to the trainable state of a model and should be exposed to an optimizer.

## Module
A reusable neural-network component that organizes Parameters, optional submodules, and a forward computation.

## Linear
A Module that applies an affine transformation, typically expressed as `y = Wx + b`, by composing primitive tensor Operations with learnable Parameters.

## ReLU
Rectified Linear Unit. An element-wise activation defined as `ReLU(x) = max(0, x)`. It introduces non-linearity without changing tensor shape or adding learnable Parameters; in MyTorch its backward passes gradients where the input is positive and blocks them elsewhere.

## Neural Network
A composition of Modules whose forward computation maps input Tensors to a prediction.
