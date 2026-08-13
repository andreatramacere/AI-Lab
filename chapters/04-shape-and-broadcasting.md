# 04 — Shape e broadcasting

## Scopo e posizione nella mappa

I primi tre capitoli hanno chiuso il ciclo che collega calcolo differenziabile, modello e aggiornamento dei parametri. Quel ciclo funziona, ma le operazioni element-wise richiedevano tensori con shape identiche.

Questo capitolo espande il core computazionale:

```text
CORE COMPUTAZIONALE
Tensor → Operation → Computational Graph → Autograd
             ↓
      SEMANTICA DELLE SHAPE
             ↓
        BROADCASTING
             ↓
            BATCH
```

Il broadcasting non è soltanto una comodità sintattica. Definisce come un'operazione mette in corrispondenza elementi appartenenti a spazi indicizzati in modo differente e come il backward deve ricondurre i gradienti alle shape originali.

Attraverseremo tre livelli:

- **forma**: stabiliremo quando due shape sono compatibili;
- **calcolo**: vedremo come un valore viene riutilizzato nel forward;
- **autograd**: capiremo perché il riutilizzo diventa una somma nel backward.

---

## 4.1 La shape è un contratto

Un `Tensor` conserva la propria shape:

```python
self.shape = _infer_shape(data)
```

Finora l'abbiamo usata soprattutto per validare gli input. In realtà, la shape definisce il dominio indicizzato su cui agisce un'operazione.

Per esempio:

```text
Tensor shape (2, 3)

indice 0        indice 1
   ↓               ↓
righe           colonne

[[x₀₀, x₀₁, x₀₂],
 [x₁₀, x₁₁, x₁₂]]
```

In un'operazione element-wise tra shape uguali, la corrispondenza è immediata:

```text
(i, j) ↔ (i, j)
```

Se le shape sono diverse, l'operazione deve stabilire se esiste una corrispondenza non ambigua. Il broadcasting è il contratto che la definisce.

---

## 4.2 Il limite precedente: uguaglianza stretta

La prima versione delle operazioni binarie richiedeva:

```python
if a.shape != b.shape:
    raise ValueError(...)
```

Questo rendeva valide espressioni come

```text
(2, 3) + (2, 3) → (2, 3)
```

ma impediva un caso fondamentale per le reti neurali:

```text
attivazioni: (batch, features)
bias:        (features,)
```

Concettualmente, vogliamo applicare lo stesso bias a ogni osservazione:

```text
[[x₀₀, x₀₁, x₀₂],       [b₀, b₁, b₂]
 [x₁₀, x₁₁, x₁₂]]   +

→

[[x₀₀+b₀, x₀₁+b₁, x₀₂+b₂],
 [x₁₀+b₀, x₁₁+b₁, x₁₂+b₂]]
```

Non vogliamo copiare manualmente `bias` in una matrice. Vogliamo che l'operazione descriva il riuso.

---

## 4.3 Regole di compatibilità

Le shape vengono confrontate allineandole da destra. Due dimensioni sono compatibili se:

```text
sono uguali
oppure
una delle due vale 1
```

Le dimensioni mancanti a sinistra vengono trattate come dimensioni di ampiezza `1`.

### Esempi compatibili

```text
      (2, 3)
         (3,)
→     (2, 3)
```

La seconda shape viene letta come `(1, 3)`: il primo asse può espandersi da `1` a `2`.

```text
      (2, 1, 4)
         (3, 4)
→     (2, 3, 4)
```

Dopo l'allineamento:

```text
(2, 1, 4)
(1, 3, 4)
-----------
(2, 3, 4)
```

Uno scalare ha shape `()` e può essere combinato con qualsiasi shape:

```text
(2, 3) * () → (2, 3)
```

### Esempio incompatibile

```text
(2, 2)
   (3,)
```

Le dimensioni finali `2` e `3` non sono uguali e nessuna vale `1`. Non esiste una regola di corrispondenza, quindi l'operazione deve fallire.

### Implementazione

In [`mytorch/operations.py`](../mytorch/operations.py), `_broadcast_shape()` traduce direttamente queste regole:

```python
def _broadcast_shape(shape_a, shape_b):
    result = []
    rank = max(len(shape_a), len(shape_b))

    padded_a = (1,) * (rank - len(shape_a)) + shape_a
    padded_b = (1,) * (rank - len(shape_b)) + shape_b

    for dim_a, dim_b in zip(padded_a, padded_b):
        if dim_a == dim_b or dim_a == 1 or dim_b == 1:
            result.append(max(dim_a, dim_b))
            continue

        raise ValueError(
            f"Shapes {shape_a} and {shape_b} are not broadcastable."
        )

    return tuple(result)
```

La funzione non esegue ancora il calcolo numerico. Determina il dominio degli indici dell'output oppure rifiuta l'operazione.

---

## 4.4 Forward: espansione logica, non copia preventiva

Per calcolare un elemento dell'output, MyTorch proietta il suo indice su ciascun operando.

Consideriamo:

```text
a.shape = (2, 3)
b.shape =    (3,)
```

Per l'indice di output `(1, 2)`:

```text
a usa (1, 2)
b usa    (2,)
```

L'indice dell'asse broadcast di `b` viene sempre riportato a zero. La funzione reale è:

```python
def _project_index(output_index, input_shape):
    offset = len(output_index) - len(input_shape)
    return tuple(
        0 if dimension == 1 else output_index[offset + axis]
        for axis, dimension in enumerate(input_shape)
    )
```

`_broadcast_binary()` costruisce l'output applicando questa proiezione a entrambi gli input:

```python
def _broadcast_binary(a, shape_a, b, shape_b, fn):
    output_shape = _broadcast_shape(shape_a, shape_b)

    def value_at(output_index):
        value_a = _get_nested(
            a, _project_index(output_index, shape_a)
        )
        value_b = _get_nested(
            b, _project_index(output_index, shape_b)
        )
        return fn(value_a, value_b)

    return _build_nested(output_shape, value_at)
```

Nel modello concettuale, l'operando più piccolo viene espanso. Nell'implementazione, MyTorch non crea prima una copia espansa: riusa lo stesso valore attraverso la proiezione degli indici mentre costruisce il risultato.

---

## 4.5 Broadcasting nelle operazioni element-wise

`Add`, `Subtract` e `Multiply` condividono ora la stessa infrastruttura. Per esempio, il forward di `Add` è:

```python
def forward(self, a, b):
    return _broadcast_binary(
        a.data, a.shape,
        b.data, b.shape,
        lambda x, y: x + y,
    )
```

La regola di forma è separata dalla regola numerica locale. Per ottenere la sottrazione o la moltiplicazione cambia la funzione finale, non il meccanismo di broadcasting.

Questo mantiene distinti due aspetti:

```text
_broadcast_binary   stabilisce quali elementi corrispondono
fn                  stabilisce come combinarli
```

---

## 4.6 Cambio di direzione: dal forward al backward

Nel forward, una dimensione di ampiezza `1` può alimentare più elementi dell'output:

```text
b₀ ─→ output[0, 0]
   └→ output[1, 0]
```

Nel backward, entrambi i cammini contribuiscono al gradiente dello stesso valore:

```text
grad_output[0, 0] ─┐
                    ├→ grad_b[0]
grad_output[1, 0] ─┘
```

Pertanto l'operazione inversa del broadcasting non è una selezione, ma una **riduzione per somma** sugli assi espansi.

Questa è la relazione centrale del capitolo:

```text
FORWARD                    BACKWARD
riuso / espansione    ↔    somma / riduzione
```

---

## 4.7 Backward dell'addizione

Senza broadcasting, per `z = a + b` vale:

```text
grad_a = grad_output
grad_b = grad_output
```

La derivata locale rimane `1`, ma la shape del contributo coincide con quella dell'output. Il contributo deve quindi essere ridotto alla shape originale di ciascun operando:

```python
def backward(self, grad_output):
    a, b = self.inputs
    output_shape = self.output.shape
    return (
        _reduce_broadcast_gradient(
            grad_output, output_shape, a.shape
        ),
        _reduce_broadcast_gradient(
            grad_output, output_shape, b.shape
        ),
    )
```

Per

```text
matrix.shape = (2, 3)
bias.shape   =    (3,)
loss = (matrix + bias).sum()
```

il gradiente in uscita è una matrice di `1`:

```text
grad_matrix = [[1, 1, 1],
               [1, 1, 1]]

grad_bias   = [1+1, 1+1, 1+1]
            = [2, 2, 2]
```

Il fattore `2` non proviene dalla derivata locale dell'addizione. Proviene dal numero di volte in cui ogni elemento del bias è stato usato.

---

## 4.8 Backward della moltiplicazione

Per `z = a * b`, prima si applica la derivata locale:

```text
grad_a_full = grad_output * b
grad_b_full = grad_output * a
```

Questi gradienti hanno la shape broadcast dell'output. Solo dopo vengono ridotti:

```python
grad_a = _reduce_broadcast_gradient(
    grad_a_full, output_shape, a.shape
)
grad_b = _reduce_broadcast_gradient(
    grad_b_full, output_shape, b.shape
)
```

L'ordine concettuale è quindi:

```text
gradiente in arrivo
        ↓
derivata locale sulla shape dell'output
        ↓
riduzione alla shape dell'input
```

Per esempio:

```text
matrix = [[1, 2, 3],
          [4, 5, 6]]
scale  = [2, 3, 4]

loss = (matrix * scale).sum()
```

si ottiene:

```text
grad_matrix = [[2, 3, 4],
               [2, 3, 4]]

grad_scale  = [1+4, 2+5, 3+6]
            = [5, 7, 9]
```

---

## 4.9 Riduzione del gradiente nell'implementazione

La funzione `_reduce_broadcast_gradient()` visita ogni indice del gradiente sulla shape di output, lo proietta sulla shape originale e accumula i contributi che finiscono sullo stesso indice:

```python
def _reduce_broadcast_gradient(grad, grad_shape, target_shape):
    if grad_shape == target_shape:
        return grad

    totals = {}

    for grad_index in _iter_indices(grad_shape):
        target_index = _project_index(
            grad_index, target_shape
        )
        totals[target_index] = (
            totals.get(target_index, 0.0)
            + _get_nested(grad, grad_index)
        )

    return _build_nested(
        target_shape,
        lambda index: totals.get(index, 0.0),
    )
```

La stessa proiezione di indici svolge due ruoli duali:

```text
forward:   molti indici di output leggono lo stesso indice di input
backward:  molti gradienti di output si sommano sullo stesso indice di input
```

Questa implementazione privilegia la leggibilità architetturale. Una libreria numerica di produzione userebbe primitive vettorizzate e riduzioni sugli assi, senza dizionari Python né visita esplicita di ogni indice.

---

## 4.10 Il caso scalare

Lo scalare ha shape `()`. Non possiede assi, quindi ogni indice dell'output viene proiettato sull'unico indice scalare, la tupla vuota `()`.

Per

```text
matrix = [[1, 2],
          [3, 4]]
scalar = 2
loss = (matrix * scalar).sum()
```

il forward produce:

```text
[[2, 4],
 [6, 8]]
```

e il backward produce:

```text
grad_matrix = [[2, 2],
               [2, 2]]

grad_scalar = 1 + 2 + 3 + 4 = 10
```

Anche qui, il gradiente scalare somma tutti i contributi generati dal suo riuso.

---

## 4.11 Relazione con `Parameter` e bias

Il broadcasting non è specifico dei parametri: si applica a qualunque `Tensor`. Diventa però particolarmente importante per i `Parameter` condivisi lungo una dimensione.

Un bias con shape `(features,)` applicato a un batch con shape `(batch, features)` è lo stesso parametro usato per ogni osservazione. Autograd deve restituire un solo gradiente con shape `(features,)`:

```text
grad_bias[j] = Σ grad_output[i, j]
               i
```

Questo collega due idee introdotte nei capitoli precedenti:

```text
Parameter
  è un Tensor appartenente allo stato apprendibile

Broadcasting
  permette di riusare lo stesso Tensor su più posizioni

Gradient accumulation
  somma i contributi prodotti da quel riuso
```

Il `Linear` corrente accetta ancora input `(in_features,)`; quindi il caso batched non è ancora esposto dal layer. Ora però l'addizione del bias possiede già la semantica necessaria per supportarlo.

---

## 4.12 Invarianti verificate dai test

I test in [`mytorch/tests.py`](../mytorch/tests.py) verificano quattro contratti:

```text
test_add_broadcast_forward_and_backward
  (2, 3) + (3,) produce (2, 3)
  e il gradiente (3,) somma lungo il primo asse

test_multiply_broadcast_backward
  applica prima la derivata locale e poi la riduzione

test_scalar_broadcast
  () si espande su tutti gli assi e riceve la somma globale

test_incompatible_broadcast_shapes
  dimensioni incompatibili producono ValueError
```

I test controllano sia i valori sia le shape implicite nelle strutture annidate. Non basta verificare il forward: un broadcasting numericamente corretto ma privo della riduzione inversa produrrebbe gradienti con forma e significato errati.

---

## 4.13 Limiti e prossimo confine

Il broadcasting è ora disponibile per:

```text
Add
Subtract
Multiply
```

Non abbiamo ancora introdotto:

- assi nominati o argomenti `axis` nelle riduzioni;
- reshape, squeeze o transpose;
- storage vettorizzato;
- `MatMul` generale;
- input batch nel layer `Linear`.

Il prossimo capitolo userà il contratto appena costruito per passare da una singola osservazione a un batch:

```text
input singolo: (in_features,)
batch:         (batch_size, in_features)
```

Questo richiederà di generalizzare `MatMul` e di scegliere con precisione l'orientamento delle dimensioni nel `Linear`. Il broadcasting del bias, invece, è già risolto.

---

## Sintesi del capitolo

```text
SHAPE
  definisce il dominio indicizzato di un Tensor
      ↓
COMPATIBILITÀ
  confronta le dimensioni allineandole da destra
      ↓
BROADCAST FORWARD
  riusa valori lungo dimensioni mancanti o di ampiezza 1
      ↓
OPERAZIONE LOCALE
  combina gli elementi messi in corrispondenza
      ↓
BACKWARD
  calcola i contributi sulla shape dell'output
      ↓
RIDUZIONE
  somma i contributi sulla shape originale di ogni input
```

La dualità fondamentale è:

```text
il forward replica logicamente un valore;
il backward somma i gradienti generati da tutte le repliche.
```

Grazie a questa regola, la semantica di Autograd rimane coerente anche quando gli operandi non hanno la stessa shape.
