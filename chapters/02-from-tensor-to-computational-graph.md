# 02 — From Tensor to Computational Graph

## Purpose

The goal of MyTorch is not to build neural networks directly.

Its first objective is to build a computational engine capable of representing values, transforming them, recording those transformations, and propagating gradients. Once this core exists, learnable parameters and neural-network modules can be built on top of it.

This chapter follows the architecture from the computational core up to the first complete neural-network model.

## Architectural Overview

```text
Tensor
  ↓
Operation
  ↓
Computational Graph
  ↓
Autograd
  ↓
Parameter
  ↓
Module
  ↓
Linear
  ↓
Neural Network
```

The map is architectural, not temporal: each concept is understood by where it sits, what it depends on, and what depends on it.

---

## 2.1 Tensor

### When

A Tensor is created whenever a value enters the MyTorch computational system.

Examples include input data, model parameters, intermediate results, and loss values.

### Where

A Tensor is the object that flows through the entire computational engine.

```text
Tensor
  ↓
Operation
  ↓
Tensor
  ↓
Operation
  ↓
Tensor
```

### How

Conceptually, a Tensor represents a portion of memory together with the context required to use it in computation.

It carries:

- `data`: the numerical values;
- `shape`: how those values are interpreted;
- `stride`: how the underlying memory is traversed, when represented explicitly by the implementation;
- `requires_grad`: whether gradients must be tracked;
- `grad`: the gradient of the loss with respect to the values stored in `data`;
- `creator`: the operation that produced the Tensor.

The key gradient relation is:

```text
grad = ∂Loss / ∂data
```

### Why

A raw array stores values. A Tensor stores values plus the context needed for differentiable computation.

The Tensor is therefore the universal value-carrying object inside MyTorch.

---

## 2.2 Operation

### When

An Operation is involved whenever one or more Tensors are transformed.

Examples include addition, multiplication, matrix multiplication, ReLU, sum, and mean.

### Where

Operations sit between Tensors.

```text
Tensor
  ↓
Operation
  ↓
Tensor
```

### How

Every Operation follows the same architectural pattern:

```text
Receive input Tensors
        ↓
Compute output values
        ↓
Create output Tensor
        ↓
Register itself as creator
```

### Why

Tensor stores state; Operation transforms state.

Separating these responsibilities keeps the computational engine modular and allows new transformations to be added without turning Tensor into a monolithic class.

---

## 2.3 Computational Graph

### When

The computational graph appears automatically as Operations are executed.

It is not a structure that the user explicitly builds.

### Where

The graph is distributed across the objects involved in computation.

Each Tensor knows its `creator`, while each Operation knows the Tensors that were used as inputs.

There is no need for a central graph object.

### How

Each Operation extends the graph by one computational step.

```text
Tensor
  ↓
Operation
  ↓
Tensor
  ↓
Operation
  ↓
Tensor
```

The graph therefore emerges from the links between Tensors and Operations.

### Why

The graph preserves the history of the forward computation. That history is what makes reverse traversal possible during gradient computation.

---

## 2.4 Autograd

### When

Autograd becomes relevant when gradients are needed, typically during training.

Pure inference does not require reverse gradient propagation.

### Where

Autograd operates on the computational graph and traverses it in the reverse direction.

```text
Forward:
A → B → C → Loss

Backward:
Loss → C → B → A
```

### How

Each Operation has two conceptual faces:

```text
forward()
backward()
```

During the forward pass, the Operation maps inputs to outputs.

During the backward pass, it maps the gradient arriving from its output to the gradients required by its inputs.

Autograd coordinates these local backward rules across the graph.

### Why

No single component knows the complete derivative of the loss with respect to every earlier value.

Each Operation knows only its local derivative. Autograd composes those local derivatives by traversing the graph backwards.

---

## 2.5 Parameter

### When

A Parameter appears when a Tensor represents a learnable quantity, such as a weight or a bias.

### Where

Parameters live inside Modules.

```text
Module
├── Parameter
├── Parameter
└── Parameter
```

### How

A Parameter is a specialized Tensor. It does not introduce a new numerical representation; it adds semantic meaning to an existing Tensor abstraction.

### Why

Not every Tensor should be optimized.

Parameter tells the framework that this particular Tensor is part of the learnable state of the model and must therefore be exposed to an optimizer.

The distinction is semantic rather than computational.

---

## 2.6 Module

### When

A Module appears when Parameters and computations must be organized into a reusable neural-network component.

Typical Modules include `Linear`, activations, sequential containers, or complete models.

### Where

Module sits between individual Parameters and the complete neural network.

```text
Parameter
   ↓
Module
   ↓
Neural Network
```

A model is therefore not just a flat collection of weights: it is a hierarchy of Modules, each of which owns or contains the Parameters relevant to its computation.

### How

A Module mainly organizes state and computation.

Conceptually:

```text
Module
├── Parameters
├── optional submodules
└── forward()
```

The Module does not introduce a new mathematical primitive. It composes objects already provided by the computational core.

### Why

Without Module, every part of the training system would need to know where every individual weight and bias is stored.

With Module, Parameters can be discovered and managed through the model hierarchy. Optimizers, serialization logic, and higher-level code can interact with the Module instead of manually tracking each learnable value.

Module therefore introduces **composition** as a first-class architectural concept.

---

## 2.7 Linear

### When

A Linear layer is used when an input vector must be transformed into a new output vector through an affine transformation.

### Where

`Linear` is the first concrete Module in the architecture.

```text
Parameter
   ↓
Module
   ↓
Linear
```

### How

A Linear layer contains learnable Parameters, typically:

```text
weight : Parameter
bias   : Parameter
```

and defines a forward computation equivalent to:

```text
y = Wx + b
```

Architecturally, however, `Linear` does not implement a new computational engine. It composes primitive Operations:

```text
input Tensor
    ↓
MatMul
    ↓
Add bias
    ↓
output Tensor
```

### Why

Linear shows the separation between primitive computation and neural-network structure.

`MatMul` and `Add` are Operations. `Linear` is a Module that composes them together with learnable Parameters.

This pattern scales naturally to more complex layers: higher-level structures are built by composing lower-level primitives rather than by creating a new computational mechanism each time.

---

## 2.8 Neural Network

### When

A Neural Network appears when multiple Modules are connected to produce a prediction for a task.

For example:

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Prediction
```

### Where

The Neural Network is the highest abstraction reached in this chapter.

```text
Tensor
  ↓
Operation
  ↓
Parameter
  ↓
Module
  ↓
Neural Network
```

### How

A neural network orchestrates the `forward()` computations of its Modules.

Each Module receives a Tensor and produces another Tensor, allowing the full model to be expressed as a composition of reusable components.

### Why

The model should define **how a prediction is produced**, not how learning itself is performed.

A Neural Network therefore organizes Modules and defines the forward flow. It does not decide whether a prediction is good or bad, and it does not update its own Parameters.

Its responsibility ends at the prediction.

---

## 2.9 Boundary of This Chapter

At this point the architecture can produce a prediction, but it does not yet form a complete learning loop.

The next layer of the map is:

```text
Prediction
   ↓
Loss
   ↓
Backward
   ↓
Optimizer
   ↓
Updated Parameters
   ↺
New Forward
```

Loss and Optimizer belong to the training loop and therefore mark the next conceptual step after the computational core and model composition established here.

---

## Chapter Summary

The architecture developed in this chapter can be read as a sequence of responsibilities:

```text
Tensor              carries values and differentiable context
Operation           transforms Tensors
Computational Graph records the history of transformations
Autograd            traverses that history backwards
Parameter           marks learnable Tensors
Module              organizes Parameters and computations
Linear              composes primitive Operations into a concrete layer
Neural Network      composes Modules into a model that produces predictions
```

No single object does everything.

The power of MyTorch comes from the relationships between these abstractions. The computational engine remains small, while increasingly complex neural-network structures emerge through composition.
