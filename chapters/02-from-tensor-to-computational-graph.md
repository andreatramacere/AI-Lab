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

Fin qui abbiamo parlato del **core computazionale**. Il core sa rappresentare valori, applicare trasformazioni e calcolare gradienti, ma non possiede ancora il concetto di modello.

Dal suo punto di vista, questi oggetti sono tutti tensori:

```text
input
valore intermedio
peso
bias
prediction
```

Le operazioni non devono sapere quale ruolo ricopra ciascun tensore. `MatMul`, per esempio, applica la stessa regola numerica e differenziale sia che un operando rappresenti dati osservati sia che rappresenti pesi apprendibili.

Questa neutralità è una proprietà desiderata del core, ma lascia aperte nuove domande:

```text
Quali valori devono essere modificati durante il training?
A quale componente appartengono?
Come possiamo recuperarli senza elencarli manualmente?
Come componiamo più trasformazioni in un unico modello?
```

Per rispondere non serve aggiungere una nuova algebra. Serve cambiare livello di astrazione e attribuire **semantica**, **ownership** e **struttura** agli oggetti già differenziabili.

```text
CORE COMPUTAZIONALE        COMPOSIZIONE DEL MODELLO
Tensor → Operation   →     Parameter → Module → Model
```

La freccia tra i due livelli non indica che `Parameter` sostituisca `Tensor`. Indica una specializzazione:

```text
Tensor
  valore che può partecipare al calcolo differenziabile
        ↓ specializzazione semantica
Parameter
  Tensor che appartiene allo stato apprendibile
        ↓ organizzazione
Module
  componente che possiede Parameter e definisce un forward
        ↓ composizione
Model
  gerarchia di Module che produce una prediction
```

Il livello superiore riusa integralmente quello inferiore. I `Parameter` attraversano le stesse `Operation`, il grafo emerge nello stesso modo e Autograd applica le stesse regole locali. Cambia il significato attribuito ad alcuni tensori e il modo in cui vengono organizzati.

Questo è quindi un cambio dall'astrazione **“come si calcola e si differenzia?”** all'astrazione **“quali valori costituiscono il modello e come sono composti?”**.

---

## 2.6 Parameter: un Tensor con semantica apprendibile

Supponiamo di costruire una trasformazione affine usando tre tensori:

```text
y = W x + b
```

Per il core computazionale, `W`, `x` e `b` sono operandi. Durante il training, però, hanno ruoli differenti:

```text
x        dato fornito al modello
W, b     stato interno che il modello deve apprendere
y        risultato temporaneo del forward
```

Autograd può calcolare gradienti rispetto a tutti gli oggetti per cui `requires_grad=True`, ma questo flag risponde soltanto alla domanda:

```text
“devo propagare e conservare il gradiente per questo Tensor?”
```

Non risponde alla domanda:

```text
“questo Tensor fa parte dello stato che l'optimizer deve aggiornare?”
```

Un input può richiedere il gradiente, per esempio per studiare la sensibilità della prediction rispetto ai dati, costruire metodi di interpretabilità o ottimizzare direttamente un input. Ciò non lo trasforma in un peso del modello.

```python
x = Tensor([2.0], requires_grad=True)
```

Qui `x.grad` verrà calcolato, ma `x` non dovrebbe comparire automaticamente in `model.parameters()`.

`Parameter` introduce precisamente la semantica mancante: identifica un tensore come parte persistente e apprendibile del modello.

```text
requires_grad=True
    richiede il calcolo del gradiente

isinstance(value, Parameter)
    identifica lo stato apprendibile posseduto dal modello
```

### Architettura

Un `Parameter` è un `Tensor` che appartiene allo stato apprendibile di un modello. La differenza è semantica, non numerica: pesi e bias partecipano alle stesse operazioni degli altri tensori, ma devono essere trovati dal modello ed esposti all'ottimizzatore.

La relazione è:

```text
ogni Parameter è un Tensor
non ogni Tensor è un Parameter
```

Essendo un `Tensor`, un `Parameter` possiede `data`, `shape`, `grad`, `requires_grad` e `creator`, e può partecipare a tutte le operazioni già implementate. Non richiede una rappresentazione numerica separata né un sistema di Autograd speciale.

Normalmente un parametro è anche un tensore foglia:

```text
parameter.creator = None
```

Non è prodotto dal forward corrente: esiste già prima del forward e viene usato per costruirlo. Il gradiente raggiunge questa foglia percorrendo a ritroso le operazioni che dipendono da essa.

Essere una foglia non è però sufficiente a renderlo un parametro. Anche `x` e `target` possono essere tensori foglia. Ancora una volta, la distinzione è il ruolo architetturale.

### Implementazione

[`mytorch/parameter.py`](../mytorch/parameter.py) rende esplicita questa relazione mediante ereditarietà:

```python
class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
```

`requires_grad=True` è necessario, ma non esaurisce il significato di `Parameter`. Un normale input può richiedere il gradiente senza essere parte dello stato apprendibile. Il tipo `Parameter` permette a `Module` di riconoscere ciò che appartiene al modello.

Durante il training, i diversi aspetti del parametro cambiano in momenti differenti:

```text
creazione
    data contiene il valore iniziale
    grad è None

forward
    data viene letta dalle Operations

backward
    grad accumula ∂loss/∂parameter
    data non cambia

optimizer.step()
    data viene aggiornata usando grad

zero_grad()
    grad torna a None
    data conserva il valore aggiornato
```

Il `Parameter` mantiene la propria identità come oggetto posseduto dal modello mentre il suo stato numerico evolve. Questo permette al `Module` e all'optimizer di conservare riferimenti allo stesso oggetto senza doverlo riscoprire dopo ogni aggiornamento.

---

## 2.7 Module: composizione e ownership

Con `Parameter` abbiamo distinto i tensori apprendibili dagli altri tensori, ma non abbiamo ancora stabilito **a quale componente appartengano** né come recuperarli quando il modello cresce.

In un modello reale, mantenere manualmente una lista separata di pesi e bias sarebbe fragile:

```text
weight_1, bias_1, weight_2, bias_2, ...
```

Ogni nuova parte del modello obbligherebbe il training loop, l'optimizer e gli strumenti di salvataggio a conoscere la sua struttura interna. Il problema non è matematico: Autograd sa già calcolare i gradienti. È un problema di organizzazione dello stato e delle responsabilità.

`Module` introduce la risposta architetturale:

```text
Parameter
    stabilisce che un Tensor è apprendibile

Module
    stabilisce chi possiede quel Parameter
    e quale calcolo viene eseguito con esso
```

Un modulo costituisce quindi un confine software attorno a stato e comportamento correlati. Può rappresentare un singolo layer, un blocco composto oppure l'intero modello. Poiché un `Module` può contenere altri `Module`, la stessa astrazione funziona a scale differenti:

```text
Model
├── Block
│   ├── Layer
│   └── Layer
└── Output Layer
```

### Relazione tra Module e layer

In MyTorch, **un layer viene implementato come una sottoclasse di `Module`**.

```text
Module
  astrazione software generale per componenti del modello
        ↓ specializzazione
Layer
  Module che realizza una trasformazione della rappresentazione
```

Per esempio:

```python
class Linear(Module):
    ...

class ReLU(Module):
    ...
```

Entrambi sono layer perché ricevono un `Tensor` e producono un nuovo `Tensor` come parte del flusso del modello. Entrambi sono implementati mediante il contratto di `Module`, in particolare definendo `forward()`.

La relazione, però, non è un'identità:

```text
ogni layer di MyTorch è un Module
non ogni Module è necessariamente un singolo layer
```

`TinyNet`, per esempio, è anch'essa una sottoclasse di `Module`, ma rappresenta l'intero modello e contiene più layer:

```text
TinyNet : Module
├── layer1 : Linear, quindi Module
├── relu   : ReLU, quindi Module
└── layer2 : Linear, quindi Module
```

Questa uniformità è ciò che rende possibile la composizione: dal punto di vista del codice, un layer elementare, un blocco di layer e un modello completo espongono tutti la stessa interfaccia `forward()`.

Questa gerarchia non coincide con il grafo computazionale. La gerarchia dei moduli descrive la struttura relativamente stabile del modello e l'ownership dei parametri; il grafo computazionale descrive invece le operazioni effettivamente eseguite durante uno specifico forward.

```text
GERARCHIA DEI MODULE             GRAFO COMPUTAZIONALE
struttura del modello            storia di una computazione
possesso dei Parameter           legami Tensor ↔ Operation
esiste prima del forward         emerge durante il forward
```

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

`Module` definisce come organizzare un componente, ma non specifica quale trasformazione debba eseguire. **`Linear` implementa un layer come sottoclasse concreta di `Module`**: assegna una precisa interpretazione matematica al contratto generico `forward()` e possiede i `Parameter` necessari a realizzarlo.

Il passaggio può essere letto così:

```text
Module
  fornisce composizione, ownership e interfaccia
        ↓
Linear : Module
  implementa un layer
  sceglie una trasformazione affine
  e la costruisce con Operations esistenti
```

`Linear` collega due spazi di rappresentazione. Riceve un certo numero di componenti, ne costruisce combinazioni pesate e produce un nuovo numero di componenti. I coefficienti di queste combinazioni non sono fissati nel codice: sono `Parameter` appresi durante il training.

Per ogni componente di output, il layer calcola una diversa combinazione di tutti gli input:

```text
input x₀ ─┬────────→ output y₀
input x₁ ─┼────────→ output y₁
input x₂ ─┴────────→ output y₂

ogni collegamento possiede un peso apprendibile
ogni output possiede inoltre un bias apprendibile
```

Il nome storico `Linear` richiede una precisazione: con il bias, la trasformazione `Wx + b` è matematicamente **affine**, non strettamente lineare. I framework mantengono comunque il nome `Linear` per questo tipo di layer.

Dal punto di vista di MyTorch, l'aspetto decisivo è che `Linear` non aggiunge una nuova primitiva al core computazionale. Compone oggetti già disponibili:

```text
Parameter weight
Parameter bias
Tensor input
Operation MatMul
Operation Add
```

In questo modo il layer ottiene automaticamente costruzione del grafo e backward dalle operazioni sottostanti. `Linear` decide **quale calcolo comporre**; non reimplementa **come differenziarlo**.

### Matematica

Per un vettore di input `x`, un layer `Linear` applica una trasformazione affine:

```text
y = W x + b
```

I due argomenti del costruttore specificano il contratto tra lo spazio di ingresso e quello di uscita:

```python
Linear(in_features, out_features)
```

```text
in_features
    numero di componenti che il layer deve ricevere

out_features
    numero di componenti che il layer deve produrre
```

Le shape attualmente supportate sono:

```text
x : (in_features,)
W : (out_features, in_features)
b : (out_features,)
y : (out_features,)
```

Ogni riga di `W` contiene i pesi di una componente di output. Per esempio:

```text
Linear(3, 2)

x.shape      = (3,)
weight.shape = (2, 3)
bias.shape   = (2,)
y.shape      = (2,)
```

Esplicitamente:

```text
y₀ = W₀₀x₀ + W₀₁x₁ + W₀₂x₂ + b₀
y₁ = W₁₀x₀ + W₁₁x₁ + W₁₂x₂ + b₁
```

Il layer trasforma quindi un punto di uno spazio a tre componenti in un punto di uno spazio a due componenti. Non esiste alcun requisito per cui `in_features` e `out_features` debbano coincidere: cambiare dimensionalità è precisamente una delle funzioni del layer.

La compatibilità richiesta è interna alla moltiplicazione:

```text
W.shape = (2, 3)
x.shape =    (3,)
                    ↑
        queste dimensioni devono coincidere
```

La dimensione condivisa viene contratta dalla somma di `MatMul`; rimane la dimensione delle righe di `W`, cioè `out_features`.

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

### Le tre shape da non confondere

Quando si parla della “shape di un layer” si possono intendere tre cose differenti:

```text
shape dell'input       (in_features,)
shape dei pesi         (out_features, in_features)
shape dell'output      (out_features,)
```

Il `Module` in sé non è un tensore e quindi non possiede una singola `shape`. Possiede invece un contratto input-output e parametri con shape determinate da quel contratto.

---

## 2.9 Neural Network: gerarchia di Module

Una rete neurale compare quando più moduli vengono composti per produrre una prediction:

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

### Perché i due `Linear` hanno shape differenti

In `TinyNet`, il primo layer espande la rappresentazione da una componente a quattro:

```text
layer1 = Linear(1, 4)

input.shape         = (1,)
layer1.weight.shape = (4, 1)
layer1.bias.shape   = (4,)
output.shape        = (4,)
```

Il secondo layer comprime la rappresentazione da quattro componenti a una:

```text
layer2 = Linear(4, 1)

input.shape         = (4,)
layer2.weight.shape = (1, 4)
layer2.bias.shape   = (1,)
output.shape        = (1,)
```

Le matrici dei pesi hanno dunque shape differenti:

```text
layer1.weight : (4, 1)
layer2.weight : (1, 4)
```

Non vengono moltiplicate direttamente tra loro. Ciascun layer moltiplica i propri pesi per il tensore che riceve. La composizione è valida perché l'output del primo ha la shape richiesta dall'input del secondo:

```text
Linear(1, 4) → output (4,) → input (4,) → Linear(4, 1)
                       shape di collegamento
```

La regola generale per comporre due layer è:

```text
Linear(n, h) → Linear(h, m)
          ↑              ↑
          └── dimensione di collegamento
```

I layer possono quindi avere matrici dei pesi differenti, ma l'interfaccia tra loro deve essere compatibile:

```text
out_features del layer precedente
                  =
in_features del layer successivo
```

Se scrivessimo invece:

```python
self.layer1 = Linear(1, 4)
self.layer2 = Linear(3, 1)
```

il primo produrrebbe `(4,)`, mentre il secondo richiederebbe `(3,)`. Il controllo in `Linear.forward()` solleverebbe quindi un `ValueError`: non esiste una connessione dimensionale valida.

### Il ruolo di ReLU tra i due layer

`ReLU` non modifica la shape:

```text
(4,) → ReLU → (4,)
```

Modifica i valori, introducendo una non linearità. Il flusso completo delle shape è:

```text
input       Linear(1,4)       ReLU       Linear(4,1)    prediction
 (1,)  ─────────────────→     (4,)  ─────────────────→    (1,)
```

Più precisamente:

```text
(1,) → Linear(1,4) → (4,) → ReLU → (4,) → Linear(4,1) → (1,)
```

Senza ReLU, la composizione di due trasformazioni affini sarebbe ancora una singola trasformazione affine:

```text
W₂(W₁x + b₁) + b₂
= (W₂W₁)x + (W₂b₁ + b₂)
```

La rappresentazione intermedia a quattro componenti acquista capacità espressiva non lineare proprio perché `ReLU` viene applicata prima del secondo layer.

### Dalle shape alla topologia del modello

La sequenza

```text
1 → 4 → 1
```

descrive la larghezza delle rappresentazioni attraversate dal modello:

```text
1 componente di input
        ↓
4 componenti nascoste
        ↓
1 componente di output
```

Il valore `4` non deriva dalla shape dei dati originali: è una scelta architetturale, detta dimensione nascosta. Stabilisce quanti valori intermedi il modello può costruire prima di produrre la prediction.

Questo oggetto svolge tre funzioni:

- definisce la topologia del modello mediante la composizione dei moduli;
- possiede indirettamente i parametri dei layer annidati;
- descrive come ottenere una predizione da un input.

Non valuta la qualità della predizione e non aggiorna i parametri. Queste responsabilità appartengono rispettivamente a loss e optimizer.

---

## 2.10 Verifica end-to-end e confine con il training loop

Il core computazionale, il modello e il training loop diventano osservabili insieme in una singola iterazione:

```python
model = TinyNet()
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

x = Tensor([2.0])
target = Tensor([4.0])

optimizer.zero_grad()
prediction = model(x)
loss = loss_fn(prediction, target)
loss.backward()
optimizer.step()
```

Nello snippet:

```text
model       istanza di TinyNet, definita nella sezione 2.9
loss_fn     istanza della Mean Squared Error di MyTorch
optimizer   SGD che gestisce i Parameter esposti dal modello
x           input con shape (1,)
target      valore atteso con shape (1,)
```

`model.parameters()` restituisce i pesi e i bias dei due layer annidati. L'optimizer conserva riferimenti a questi stessi oggetti `Parameter`: quando esegue `step()`, modifica quindi lo stato appartenente a `model`.

Queste righe non rappresentano un'unica operazione monolitica. Coordinano componenti con responsabilità e stati differenti:

```text
optimizer.zero_grad()            delimita una nuova raccolta di gradienti
model(x)                         esegue il forward e costruisce il grafo del modello
loss_fn(prediction, target)      prolunga il grafo fino a un obiettivo scalare
loss.backward()                  percorre il grafo e accumula gradienti
optimizer.step()                 modifica lo stato apprendibile
```

### 1. Stato iniziale

Prima del forward esistono già:

```text
model
├── layer1.weight : Parameter
├── layer1.bias   : Parameter
├── layer2.weight : Parameter
└── layer2.bias   : Parameter

x       : Tensor di input
target  : Tensor osservato
```

I parametri contengono i valori correnti del modello. `x` e `target` contengono i dati dell'esempio; nel training corrente non richiedono gradienti.

`optimizer.zero_grad()` non modifica i valori dei parametri. Cancella soltanto i gradienti accumulati in un'iterazione precedente:

```text
parameter.data    rimane invariato
parameter.grad    diventa None
```

Questa chiamata stabilisce il confine tra due raccolte di contributi al gradiente. Il motivo dell'accumulazione e le possibili strategie alternative saranno sviluppati nel capitolo 3.

### 2. Il modello produce la prediction

La chiamata

```python
prediction = model(x)
```

attraversa la gerarchia dei `Module`:

```text
x
↓
layer1: weight₁ @ x + bias₁
↓
ReLU
↓
layer2: weight₂ @ hidden + bias₂
↓
prediction
```

A livello architetturale diciamo che il modello termina alla prediction perché il suo contratto è:

```text
input → prediction
```

A livello del grafo computazionale, invece, la prediction conserva i collegamenti alle operazioni che l'hanno generata. Attraverso quei collegamenti è ancora possibile risalire ai parametri:

```text
prediction.creator
        ↓
operazioni del secondo layer
        ↓
attivazione intermedia
        ↓
operazioni del primo layer
        ↓
Parameter
```

Quindi “il modello termina alla prediction” descrive un **confine di responsabilità**, non un'interruzione del grafo.

### 3. La loss prolunga il grafo

La chiamata

```python
loss = loss_fn(prediction, target)
```

non assegna semplicemente un numero alla prediction. Costruisce nuove operazioni differenziabili a valle del modello. La `MSELoss` di MyTorch è composta da primitive già presenti nel core:

```python
error = prediction - target
squared_error = error ** 2
total = squared_error.sum()
return total * scale
```

Il grafo completo diventa:

```text
Parameter
    ↓
operazioni del modello
    ↓
prediction ─┐
            ├→ Subtract → Power → Sum → Multiply → loss
target ─────┘                                  ↑
                                              scale
```

`target` e `scale` partecipano al calcolo numerico, ma non richiedono gradienti. La prediction, invece, ha `requires_grad=True` perché dipende dai `Parameter`. Anche la loss risultante richiede quindi il gradiente.

La loss è scalare. Questo fornisce una singola quantità rispetto alla quale esprimere tutte le sensibilità:

```text
∂loss/∂weight₁
∂loss/∂bias₁
∂loss/∂weight₂
∂loss/∂bias₂
```

### 4. Il backward interroga la storia del forward

La chiamata

```python
loss.backward()
```

parte dal seed:

```text
∂loss/∂loss = 1
```

e attraversa il grafo in direzione opposta:

```text
loss
  ↓ backward delle operazioni della loss
prediction.grad
  ↓ backward del secondo Linear
gradienti di weight₂, bias₂ e hidden
  ↓ backward di ReLU
gradiente prima dell'attivazione
  ↓ backward del primo Linear
gradienti di weight₁ e bias₁
```

Ogni `Operation.backward()` riceve `grad_output`, lo combina con la propria derivata locale e restituisce un gradiente per ciascun input. `Tensor.backward()` coordina ricorsivamente il passaggio da un'operazione alla precedente.

Al termine del backward:

```text
parameter.data    contiene ancora gli stessi valori del forward
parameter.grad    contiene ∂loss/∂parameter
```

Questo punto è essenziale: **il backward non addestra ancora il modello** nel senso di modificarne i pesi. Calcola l'informazione necessaria a una successiva regola di ottimizzazione.

### 5. L'optimizer muta i parametri

La chiamata

```python
optimizer.step()
```

non percorre il grafo e non calcola derivate. Legge i gradienti già presenti nei `Parameter` e applica la regola SGD:

```text
parameter.data ← parameter.data - learning_rate · parameter.grad
```

La separazione è quindi:

```text
Autograd
    determina in quale direzione e con quale sensibilità varia la loss

Optimizer
    decide come usare quell'informazione per modificare i parametri
```

Lo stesso grafo e gli stessi gradienti potrebbero essere utilizzati da una diversa regola di ottimizzazione. Analogamente, SGD non ha bisogno di conoscere se il gradiente provenga da una MSE, da un'altra loss o da una particolare architettura.

### 6. Perché serve un nuovo forward

Dopo `optimizer.step()`, la prediction già calcolata non cambia retroattivamente. È un `Tensor` contenente il risultato ottenuto con i vecchi valori dei parametri.

```text
vecchi parametri → vecchia prediction → loss → gradienti
                                         ↓
                                  aggiornamento
                                         ↓
                                  nuovi parametri
```

Per osservare l'effetto dell'aggiornamento bisogna eseguire nuovamente:

```python
new_prediction = model(x)
```

Il nuovo forward legge i valori aggiornati dei `Parameter` e costruisce un nuovo grafo:

```text
nuovi parametri → nuovo forward → nuova prediction → nuova loss
```

L'apprendimento diventa osservabile solo confrontando il comportamento del modello prima e dopo l'aggiornamento. Una singola iterazione non garantisce in generale che la loss diminuisca: ciò dipende dalla regola di ottimizzazione, dal learning rate e dalla geometria locale dell'obiettivo.

### Confini di responsabilità

L'intera sequenza può essere riletta come una tabella di contratti:

| Componente | Legge | Produce o modifica | Non decide |
|---|---|---|---|
| `Model` | input e parametri | prediction e grafo del forward | qualità della prediction |
| `Loss` | prediction e target | obiettivo scalare e prosecuzione del grafo | aggiornamento dei parametri |
| `Autograd` | grafo e regole locali | gradienti | politica di ottimizzazione |
| `Optimizer` | parametri e gradienti | nuovi valori dei parametri | struttura del modello e loss |

Questi confini permettono di sostituire un componente senza riscrivere gli altri: una nuova loss può valutare lo stesso modello; un nuovo optimizer può usare gli stessi gradienti; una nuova architettura può partecipare allo stesso training loop.

Il modello termina dunque concettualmente alla `Prediction`. La loss collega quella prediction a un criterio; Autograd traduce il criterio in gradienti; l'optimizer traduce i gradienti in una mutazione dello stato apprendibile. Il capitolo 3 analizzerà quantitativamente questo ciclo mediante un passo SGD completo.

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
