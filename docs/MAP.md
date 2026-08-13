# AI Lab Map

The map is architectural, not temporal. It shows where each concept sits and how the layers depend on one another; it does not prescribe a rigid learning order.

```text
DATA
  ↓
TENSORS
  ↓
OPERATIONS
  ↓
COMPUTATIONAL GRAPH
  ↓
AUTOGRAD
  ↓
PARAMETERS
  ↓
MODULES / LAYERS
  ↓
MODEL
  ↓
PREDICTION
  ↓
LOSS
  ↓
BACKWARD
  ↓
GRADIENTS
  ↓
OPTIMIZER
  ↓
PARAMETER UPDATE
  ↺ new forward
```

## Current architectural layers

```text
COMPUTATIONAL CORE
Tensor → Operation → Computational Graph → Autograd

MODEL COMPOSITION
Parameter → Module → Linear → Neural Network

TRAINING LOOP
Prediction → Loss → Backward → Optimizer → Parameter Update
```

## Longer-range map

```text
COMPUTATIONAL CORE
  ↓
NEURAL NETWORKS
  ↓
TRAINING
  ↓
SCALABILITY
Broadcasting → Batch → General MatMul → Vectorization
  ↓
MODERN DEEP LEARNING
Initialization → Normalization → Regularization → Adam
  ↓
SEQUENCES
Embedding → Token Representation
  ↓
ATTENTION
Q/K/V → Attention → Self-Attention → Multi-Head Attention
  ↓
TRANSFORMER
Residual → Normalization → Feed Forward → Transformer Block
  ↓
LANGUAGE MODEL
Tokenizer → Causal Mask → Next-token Prediction → Generation
  ↓
DOMAIN EXPERT MODEL
Domain Corpus → Adaptation / Training → Evaluation
```
