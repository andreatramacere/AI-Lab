02 From Tensor To Computational Graph

Purpose

The goal of MyTorch is not to build neural networks directly.

Its first objective is to build a computational engine capable of representing values, transforming them, recording those transformations, and propagating gradients.

Everything else in the framework is built on top of this core.

⸻

Architectural Overview

Tensor
    │
    ▼
Operation
    │
    ▼
Computational Graph
    │
    ▼
Autograd
    │
    ▼
Parameter

These five concepts form the computational foundation of MyTorch.

Understanding them means understanding how every neural network library works internally.

⸻

2.1 Tensor

When

A Tensor is created whenever a value enters the computational system.

Examples include:

* input data
* model parameters
* intermediate results
* loss values

Every quantity that participates in computation is represented as a Tensor.

⸻

Where

A Tensor is the object that flows through the entire computational engine.

Tensor
   ↓
Operation
   ↓
Tensor
   ↓
Operation
   ↓
Tensor

Everything operates on Tensors and produces new Tensors.

⸻

How

A Tensor stores both numerical data and the information required for automatic differentiation.

Conceptually, a Tensor contains:

* data
* shape
* stride
* requires_grad
* grad
* creator

data

The numerical values.

shape

How those values are interpreted.

stride

How memory is traversed.

Depending on the implementation, stride may be stored explicitly or derived from the underlying storage.

requires_grad

Whether gradients must be computed.

grad

The gradient of the loss with respect to the Tensor’s data.

grad = ∂Loss / ∂data

creator

The operation that produced this Tensor.

⸻

Why

A raw array only stores numbers.

A Tensor stores numbers together with the information necessary for gradient-based computation.

It is the universal object that travels inside MyTorch.

⸻

2.2 Operation

When

Every transformation of one or more Tensors is an Operation.

Examples include:

* addition
* multiplication
* matrix multiplication
* ReLU
* sum
* mean

⸻

Where

Operations sit between Tensors.

Tensor
   │
   ▼
Operation
   │
   ▼
Tensor

⸻

How

Every Operation performs the same conceptual sequence.

Receive input Tensors
        ↓
Compute new values
        ↓
Create output Tensor
        ↓
Register itself as creator

⸻

Why

Tensor stores state.

Operation transforms state.

Keeping these responsibilities separate makes the architecture modular and extensible.

⸻

2.3 Computational Graph

When

A computational graph appears automatically as Operations are executed.

It is not built explicitly by the user.

⸻

Where

The graph is distributed.

Each Tensor knows its creator.

Each Operation knows its inputs.

There is no central graph object.

⸻

How

Each Operation contributes one node.

Tensor
    ↓
Operation
    ↓
Tensor
    ↓
Operation
    ↓
Tensor

As execution progresses, the graph grows naturally.

⸻

Why

The graph records the complete history of the forward computation.

This history will later be traversed during backpropagation.

⸻

2.4 Autograd

When

Autograd is used during training.

It is unnecessary during pure inference.

⸻

Where

Autograd operates on top of the computational graph.

It traverses the graph in reverse order.

Forward
A → B → C → Loss
Backward
Loss → C → B → A

⸻

How

Each Operation implements two conceptual behaviors.

Forward()
Backward()

During the forward pass:

input
   ↓
output

During the backward pass:

∂output
     ↓
∂input

Autograd coordinates this reverse traversal.

⸻

Why

No component knows the complete derivative.

Each Operation contributes only its local derivative.

Autograd combines these local derivatives to compute gradients throughout the graph.

⸻

2.5 Parameter

When

A Parameter appears whenever a Tensor represents a learnable quantity.

Typical examples include:

* weights
* biases

⸻

Where

Parameters belong to Modules.

Module
 ├── Parameter
 ├── Parameter
 └── Parameter

⸻

How

A Parameter is a specialized Tensor.

It inherits the computational capabilities of Tensor while adding semantic meaning.

⸻

Why

Not every Tensor should be optimized.

A Parameter explicitly identifies the values that an optimizer must update during training.

The distinction is semantic rather than computational.

⸻

Chapter Summary

The computational core of MyTorch consists of five concepts.

Tensor
    │
    ▼
Operation
    │
    ▼
Computational Graph
    │
    ▼
Autograd
    │
    ▼
Parameter

Tensor carries information.

Operation transforms information.

Operations collectively build the Computational Graph.

Autograd traverses that graph backwards to compute gradients.

Parameters identify which Tensors should be updated during learning.

Everything that follows in MyTorch—Modules, Layers, Neural Networks, Losses and Optimizers—is built on top of this computational foundation.