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
A reusable neural-network component that organizes Parameters, optional submodules, and a forward computation. `named_parameters()` exposes learnable state through hierarchical ownership paths such as `layer1.weight`, while `parameters()` exposes the same Parameter objects without names for optimizers.

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
A conceptual computational unit of a layer, distinct from the scalar value it produces. In a fully connected layer, unit `j` is associated with parameters `W[j, :]` and `b[j]` and, for a specific input, computes the scalar pre-activation `z_j = Σ_i W_ji h_i + b_j`, followed when applicable by `h_j = a(z_j)`. The neuron belongs to the architectural and operational description of the layer; `z_j` and `h_j` are transient execution values. In a tensor-based implementation, neurons are normally not separate software objects: one layer computes all their values together and collects them in a Tensor.

## Fully Connected Layer
A layer in which every output unit receives every input coordinate. Its weight matrix has shape `(out_features, in_features)`: row `j` contains all weights entering output unit `j`, and element `W_ji` is the connection weight from input coordinate `i` to unit `j`.

## Pre-activation
The intermediate Tensor `z` produced by a parametrized transformation before applying its activation function, for example `z = Wh + b`.

## Post-activation
The Tensor `h = a(z)` produced after applying an activation function. It is commonly the hidden representation passed to the next layer.

## Output Head
The final model component that maps the last hidden representation into the output space required by the task. The hidden space has internally learned coordinates, whereas the prediction's shape and semantics are determined by the target, such as regression values, class scores, or vocabulary logits.

## Output Dimension
The number of coordinates produced by an output head. For a linear head mapping hidden dimension `h` to output dimension `o`, the weight shape is `(o, h)` and the mapping is `(o, h) @ (h,) → (o,)`. It is determined by the task and need not equal the input dimension.

## Prediction
Il Tensor esposto al confine di output di un modello. Dipende dai Parameter correnti attraverso il grafo computazionale, mentre la sua shape e la sua semantica sono determinate dal task e dal contratto della loss o del post-processing successivo.

## Target
Il valore di riferimento fornito dal dataset o dal task rispetto al quale viene valutata una prediction. Nel training supervisionato considerato è esterno al modello e non richiede gradienti.

## Loss
Un criterio differenziabile che combina una prediction con un target e produce l'obiettivo da ottimizzare, normalmente un Tensor scalare. Appartiene al sistema di training, non al modello, e prolunga il grafo computazionale a valle della prediction.

## Gradient
La derivata di un obiettivo scalare rispetto a un Tensor o a un Parameter. In MyTorch, Autograd calcola i contributi durante il backward e li accumula nel campo `grad` dell'oggetto.

## Optimizer
Un componente del sistema di training che legge i gradienti dei Parameter gestiti e applica una regola di aggiornamento ai loro dati numerici. Non calcola i gradienti, non sceglie la loss e non esegue il forward del modello.

## SGD
Stochastic Gradient Descent. Una regola di ottimizzazione che aggiorna un Parameter `θ` secondo `θ ← θ - η ∂L/∂θ`, dove `L` è la loss e `η` il learning rate. Nel loop corrente di MyTorch, “stochastic” indica che l'aggiornamento deriva da un esempio di training alla volta.

## Learning Rate
Lo scalare `η` che controlla l'ampiezza dell'aggiornamento prodotto da un optimizer. In SGD scala il passo compiuto nella direzione opposta al gradiente.

## Gradient Accumulation
La raccolta intenzionale dei contributi di gradiente prodotti da più backward prima di un passo dell'optimizer. Si basa sulla somma dei gradienti anziché sulla loro sovrascrittura e richiede `zero_grad()` soltanto all'inizio di una nuova finestra di accumulazione.
