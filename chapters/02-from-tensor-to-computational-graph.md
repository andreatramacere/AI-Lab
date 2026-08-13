# 02 — Dal Tensor al grafo computazionale

## Scopo e posizione nella mappa

MyTorch non nasce per costruire direttamente reti neurali. Il suo primo compito è fornire un motore capace di rappresentare valori, trasformarli, conservare la storia delle trasformazioni e propagare gradienti.

Questo capitolo attraversa due livelli architetturali:

```text
CORE COMPUTAZIONALE
Tensor → Operation → Computational Graph → Autograd

COMPOSIZIONE DEL MODELLO
Parameter → Module → Linear → Neural Network
```

Il primo livello rende possibile il calcolo differenziabile; il secondo usa quel meccanismo per organizzare quantità apprendibili e costruire un modello. La mappa è spaziale, non temporale: per ogni oggetto ci chiediamo da cosa dipende, che responsabilità possiede e chi dipende da esso.

Il training loop rimane fuori dal confine principale del capitolo:

```text
Prediction → Loss → Backward → Gradients → Optimizer → Parameter Update
```

Ne incontreremo alcuni elementi per verificare che l'architettura funzioni, senza confonderli con il modello.

---

## 2.1 Tensor: il valore che attraversa il sistema

### Architettura

Un `Tensor` è l'oggetto che fluisce attraverso tutto il motore:

```text
Tensor → Operation → Tensor → Operation → Tensor
```

Un array contiene valori. Un `Tensor` di MyTorch contiene valori e il contesto minimo necessario al calcolo differenziabile:

- `data`: scalare o liste annidate di valori numerici;
- `shape`: interpretazione dimensionale dei dati;
- `requires_grad`: indica se il gradiente deve essere propagato;
- `grad`: gradiente accumulato rispetto alla quantità scalare finale;
- `creator`: operazione che ha prodotto il tensore, oppure `None` per un tensore foglia.

Nel codice attuale non esistono ancora storage lineare e stride espliciti: `data` usa scalari e liste Python annidate. È una scelta di implementazione temporanea, non una diversa definizione architetturale di tensore.

### Implementazione

In [`mytorch/tensor.py`](../mytorch/tensor.py), il costruttore materializza esattamente questo stato:

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.shape = _infer_shape(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = None
```

Gli operatori del `Tensor` non implementano direttamente l'algebra. Delegano alle corrispondenti `Operation`:

```python
def __mul__(self, other):
    from operations import Multiply
    return Multiply()(self, other)

def __matmul__(self, other):
    from operations import MatMul
    return MatMul()(self, other)
```

Questo dettaglio rende visibile la separazione delle responsabilità: `Tensor` trasporta stato, `Operation` esegue e registra una trasformazione.

### Matematica

Se `L` è lo scalare da cui parte la retropropagazione, il campo `x.grad` rappresenta

```text
∂L / ∂x
```

Non è una proprietà isolata di `x`: dipende dal percorso che collega `x` a `L` nel grafo.

---

## 2.2 Operation: una trasformazione con memoria locale

### Architettura

Una `Operation` riceve uno o più tensori, calcola nuovi dati e restituisce un nuovo tensore:

```text
Tensor di input
      ↓
   Operation
      ↓
Tensor di output
```

Oltre al calcolo numerico, conserva ciò che servirà durante il backward. L'interfaccia comune vive in [`mytorch/operations.py`](../mytorch/operations.py):

```python
class Operation:
    def __call__(self, *inputs):
        self.inputs = inputs
        data = self.forward(*inputs)

        requires_grad = any(tensor.requires_grad for tensor in inputs)

        result = Tensor(data, requires_grad=requires_grad)
        result.creator = self
        self.output = result

        return result
```

La chiamata svolge quattro operazioni architetturali:

1. conserva i tensori di input;
2. delega il calcolo locale a `forward()`;
3. crea il tensore di output;
4. collega l'output all'operazione mediante `creator`.

`forward()` e `backward()` hanno responsabilità diverse:

```python
def forward(self, *inputs):
    raise NotImplementedError

def backward(self, grad_output):
    raise NotImplementedError
```

Il primo calcola valori; il secondo applica la regola differenziale locale.

### Esempio: moltiplicazione

Per `z = a * b`, la matematica locale è:

```text
∂z/∂a = b
∂z/∂b = a
```

Queste sono le derivate di `z` rispetto ai suoi input, ma non sono ancora i gradienti che interessano al training. Se una loss scalare `L` dipende da `z`, vogliamo calcolare:

```text
∂L/∂a
∂L/∂b
```

La chain rule fornisce:

```text
∂L/∂a = (∂L/∂z)(∂z/∂a) = (∂L/∂z)b
∂L/∂b = (∂L/∂z)(∂z/∂b) = (∂L/∂z)a
```

Nell'interfaccia di MyTorch:

```text
grad_output = ∂L/∂z
```

Di conseguenza, `Multiply.backward()` non restituisce semplicemente `b` e `a`. Combina il gradiente ricevuto da valle con le derivate locali:

```python
class Multiply(Operation):
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
```

La riduzione finale serve quando uno degli input è stato riutilizzato mediante broadcasting; questo meccanismo sarà analizzato nel capitolo 4. Senza broadcasting, `grad_a_full` e `grad_b_full` hanno già le shape corrette.

L'operazione non conosce l'intero modello né la formula completa della loss. Riceve da valle l'effetto della parte successiva del grafo e conosce la propria trasformazione locale. Queste due informazioni sono sufficienti.

---

## 2.3 Il grafo computazionale emerge dalle relazioni

### Architettura

MyTorch non possiede un oggetto centrale `Graph`. Il grafo è distribuito tra:

- il `creator` conservato da ogni tensore prodotto;
- gli `inputs` conservati da ogni operazione.

Per l'espressione

```python
loss = (a * b).sum()
```

si forma dinamicamente la struttura:

```text
a ─┐
   ├→ Multiply → risultato intermedio → Sum → loss
b ─┘
```

Il forward costruisce il grafo mentre esegue il calcolo. Non è necessaria una fase separata di dichiarazione.

### Perché il grafo è necessario

Il valore di `loss` non contiene da solo informazioni sufficienti per ricostruire come dipende da `a` e `b`. I collegamenti tra tensori e operazioni preservano la storia causale del calcolo. Autograd usa quella storia in direzione inversa.

Questa distinzione è importante:

```text
forward   produce valori e costruisce il grafo
backward  percorre il grafo e produce gradienti
```

---

## 2.4 Autograd: composizione delle derivate locali

### Matematica

Se

```text
x → f → y → g → L
```

la chain rule fornisce

```text
∂L/∂x = (∂L/∂y)(∂y/∂x)
```

Ogni `Operation` calcola il proprio fattore locale. Autograd coordina la loro composizione lungo il grafo.

### Derivata locale e gradiente non sono la stessa cosa

Consideriamo:

```text
a = [2, 3]
b = [4, 5]
z = a * b = [8, 15]
L = z.sum() = 23
```

Il grafo è:

```text
a ─┐
   ├→ Multiply → z → Sum → L
b ─┘
```

La derivata locale di `Multiply` rispetto ad `a` è `b`:

```text
∂z/∂a = [4, 5]
```

Il backward, però, parte dalla loss. `Sum.backward()` comunica a `Multiply`:

```text
grad_output = ∂L/∂z = [1, 1]
```

`Multiply.backward()` compone le due informazioni:

```text
∂L/∂a = grad_output * b
       = [1, 1] * [4, 5]
       = [4, 5]

∂L/∂b = grad_output * a
       = [1, 1] * [2, 3]
       = [2, 3]
```

In questo esempio il gradiente rispetto ad `a` coincide numericamente con `b` soltanto perché `Sum` invia un vettore di `1`.

Se invece:

```text
L = sum(z²)
```

allora l'operazione `Power` invia:

```text
grad_output = ∂L/∂z = 2z
```

e `Multiply.backward()` restituisce:

```text
∂L/∂a = (2z)b
∂L/∂b = (2z)a
```

La distinzione generale è:

```text
derivata locale
    descrive come l'output dell'Operation cambia rispetto a un input

grad_output
    descrive come la loss cambia rispetto all'output dell'Operation

grad_input
    combina le due informazioni mediante la chain rule
```

In forma compatta:

```text
grad_input = grad_output × derivata locale
```

Per tensori multidimensionali questa espressione rappresenta, più precisamente, un prodotto vettore-Jacobiana. MyTorch non costruisce la Jacobiana completa: ogni `backward()` calcola direttamente il suo effetto sul gradiente ricevuto. È questa scelta che rende praticabile la reverse-mode autodiff.

### Implementazione

Nel codice attuale il coordinamento è ricorsivo e risiede in `Tensor.backward()`:

```python
def backward(self, grad=None):
    if not self.requires_grad:
        return

    if grad is None:
        if self.shape != ():
            raise RuntimeError(
                "backward() without an explicit grad is allowed only "
                "for scalar tensors."
            )
        grad = 1.0

    self._accumulate_grad(grad)

    if self.creator is None:
        return

    input_grads = self.creator.backward(grad)

    for tensor, tensor_grad in zip(self.creator.inputs, input_grads):
        if tensor.requires_grad:
            tensor.backward(tensor_grad)
```

La chiamata senza argomento è ammessa solo su uno scalare e semina il backward con `1.0`, perché

```text
∂L/∂L = 1
```

### Accumulazione dei gradienti

Un tensore può contribuire alla stessa loss attraverso più rami. Per esempio:

```python
x = Tensor([2.0, 3.0], requires_grad=True)
y = x + x
loss = y.sum()
loss.backward()
```

Qui ogni elemento di `x` contribuisce due volte, quindi `x.grad == [2.0, 2.0]`. Per questo `_accumulate_grad()` somma i contributi invece di sovrascriverli.

### Limite attuale

Il backward ricorsivo è sufficiente per il laboratorio corrente, ma non implementa ancora un ordinamento topologico esplicito. Questa limitazione appartiene all'implementazione di Autograd, non alla sua responsabilità architetturale.

---

## 2.5 Cambio di livello: dal calcolo al modello

Fin qui abbiamo parlato del **core computazionale**. Ora cambiamo livello di astrazione: non aggiungiamo nuove regole di derivazione, ma attribuiamo semantica e struttura ai tensori differenziabili.

```text
CORE COMPUTAZIONALE        COMPOSIZIONE DEL MODELLO
Tensor → Operation   →     Parameter → Module → Model
```

---

## 2.6 Parameter: un Tensor con semantica apprendibile

### Architettura

Un `Parameter` è un `Tensor` che appartiene allo stato apprendibile di un modello. La differenza è semantica, non numerica: pesi e bias partecipano alle stesse operazioni degli altri tensori, ma devono essere trovati dal modello ed esposti all'ottimizzatore.

### Implementazione

[`mytorch/parameter.py`](../mytorch/parameter.py) rende esplicita questa relazione mediante ereditarietà:

```python
class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
```

`requires_grad=True` è necessario, ma non esaurisce il significato di `Parameter`. Un normale input può richiedere il gradiente senza essere parte dello stato apprendibile. Il tipo `Parameter` permette a `Module` di riconoscere ciò che appartiene al modello.

---

## 2.7 Module: composizione e ownership

### Architettura

Un `Module` organizza:

```text
Module
├── Parameter posseduti direttamente
├── Module annidati
└── forward()
```

Non introduce una nuova primitiva matematica. Introduce ownership, composizione e un'interfaccia uniforme per il forward.

### Implementazione

In [`mytorch/module.py`](../mytorch/module.py), `__call__` rende un modulo invocabile e `parameters()` attraversa ricorsivamente la gerarchia:

```python
class Module:
    def __call__(self, *inputs):
        return self.forward(*inputs)

    def parameters(self):
        params = []

        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())

        return params
```

L'ottimizzatore può così ricevere `model.parameters()` senza conoscere nomi, numero o posizione dei singoli pesi.

Il meccanismo corrente scopre parametri e sottomoduli assegnati direttamente come attributi. Contenitori generici, liste di moduli, serializzazione e gestione dello stato non sono ancora implementati.

---

## 2.8 Linear: una struttura costruita con primitive esistenti

### Matematica

Per un vettore di input `x`, un layer lineare applica la trasformazione affine

```text
y = W x + b
```

con forme attualmente supportate:

```text
x : (in_features,)
W : (out_features, in_features)
b : (out_features,)
y : (out_features,)
```

### Architettura e implementazione

`Linear` è un `Module` che possiede due `Parameter` e compone `MatMul` con `Add`. Il codice reale in [`mytorch/layers.py`](../mytorch/layers.py) è:

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        # ... inizializzazione dei valori ...
        self.weight = Parameter(weights)
        self.bias = Parameter(biases)

    def forward(self, x):
        if x.shape != (self.in_features,):
            raise ValueError(
                f"Linear expected input shape ({self.in_features},), "
                f"received {x.shape}."
            )

        return self.weight @ x + self.bias
```

Nel grafo, l'ultima riga diventa:

```text
weight ─┐
        ├→ MatMul ─┐
input  ─┘          ├→ Add → output
bias ──────────────┘
```

`Linear` non implementa un proprio backward. Non ne ha bisogno: `MatMul` e `Add` hanno già le regole locali necessarie, e Autograd compone i gradienti. È il primo esempio completo del principio secondo cui strutture di livello superiore emergono componendo primitive inferiori.

---

## 2.9 Neural Network: gerarchia di Module

Una rete neurale compare quando più moduli vengono composti per produrre una predizione:

```text
Input → Linear → ReLU → Linear → Prediction
```

[`mytorch/main.py`](../mytorch/main.py) contiene un modello concreto:

```python
class TinyNet(Module):
    def __init__(self):
        self.layer1 = Linear(1, 4)
        self.relu = ReLU()
        self.layer2 = Linear(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

Questo oggetto svolge tre funzioni:

- definisce la topologia del modello mediante la composizione dei moduli;
- possiede indirettamente i parametri dei layer annidati;
- descrive come ottenere una predizione da un input.

Non valuta la qualità della predizione e non aggiorna i parametri. Queste responsabilità appartengono rispettivamente a loss e optimizer.

---

## 2.10 Verifica end-to-end e confine con il training loop

Il core e il modello diventano osservabili insieme in una singola sequenza:

```python
prediction = model(x)
loss = loss_fn(prediction, target)
loss.backward()
optimizer.step()
```

È essenziale leggere le quattro righe ai corretti livelli di astrazione:

```text
model(x)                         forward del modello
loss_fn(prediction, target)      costruzione dell'obiettivo scalare
loss.backward()                  autograd sul grafo
optimizer.step()                 aggiornamento dello stato apprendibile
```

La loss di MyTorch è a sua volta composta da operazioni primitive:

```python
error = prediction - target
squared_error = error ** 2
total = squared_error.sum()
return total * scale
```

Questo conferma che il grafo non si arresta ai confini dei moduli: registra tutte le operazioni differenziabili che collegano i parametri alla loss.

Il modello termina concettualmente alla `Prediction`. Loss e optimizer aprono il livello successivo della mappa, il **training loop**.

---

## Sintesi del capitolo

```text
Tensor
  porta valori e contesto differenziabile
    ↓
Operation
  trasforma tensori e conserva la regola locale
    ↓
Computational Graph
  emerge dai collegamenti creati durante il forward
    ↓
Autograd
  percorre il grafo a ritroso e compone le derivate locali
    ↓
Parameter
  attribuisce semantica apprendibile a un Tensor
    ↓
Module
  organizza parametri, sottomoduli e forward
    ↓
Linear
  compone Parameter, MatMul e Add
    ↓
Neural Network
  compone Module e produce una Prediction
```

Il punto centrale non è la complessità dei singoli oggetti, ma la precisione dei loro confini. Il calcolo differenziabile appartiene al core; l'ownership dei pesi e la composizione appartengono al modello; la valutazione e l'aggiornamento appartengono al training loop. MyTorch rende queste separazioni visibili perché costruisce ogni livello usando esplicitamente quello sottostante.
