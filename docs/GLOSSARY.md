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

## Hidden Representation
An intermediate Tensor value at an internal boundary of a model's forward computation, produced for a specific input and the current Parameters. It is not a layer, the collection of hidden layers, or the persistent state of those layers. It is input-dependent and transient; its coordinates are shaped indirectly by the training objective and need not be individually interpretable.

## Hidden Layer
A model component located between the input and output boundaries. It can produce a hidden representation but is not itself that representation. Saying that a network has hidden layers describes architecture; giving `h` for an input describes one execution.

## Hidden Dimension
The number of components in a hidden representation. It is an architectural choice that controls the width of that internal representation.

## Hidden State
In recurrent architectures, an internal representation passed from one sequence step to the next, such as `h_t = f(x_t, h_{t-1}; θ)`. This specialized temporal meaning does not apply to every hidden representation in a feed-forward network.

## Neuron
A single computational unit or output coordinate of a layer. For a fully connected layer, unit `j` computes a scalar pre-activation such as `z_j = Σ_i W_ji h_i + b_j`. The values of all units are normally collected in one activation Tensor; a neuron is not usually a separate Tensor object.

## Pre-activation
The intermediate Tensor `z` produced by a parametrized transformation before applying its activation function, for example `z = Wh + b`.

## Post-activation
The Tensor `h = a(z)` produced after applying an activation function. It is commonly the hidden representation passed to the next layer.

## Output Head
The final model component that maps the last hidden representation into the output space required by the task. The hidden space has internally learned coordinates, whereas the prediction's shape and semantics are determined by the target, such as regression values, class scores, or vocabulary logits.
