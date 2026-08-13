# Mappa di AI Lab

La mappa è architetturale, non temporale. Mostra dove si colloca ogni concetto, da cosa dipende e quali componenti dipendono da esso; non prescrive un ordine di apprendimento rigido.

```text
DATI
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
MODULES / LAYER
  ↓
MODELLO
  ↓
PREDICTION
  ↓
LOSS
  ↓
BACKWARD
  ↓
GRADIENTI
  ↓
OPTIMIZER
  ↓
AGGIORNAMENTO DEI PARAMETRI
  ↺ nuovo forward
```

## Livelli architetturali consolidati

```text
CORE COMPUTAZIONALE
Tensor → Operation → Computational Graph → Autograd

COMPOSIZIONE DEL MODELLO
Parameter → Module → Linear → Neural Network

TRAINING LOOP
Prediction → Loss → Backward → Gradients → Optimizer
     ↑                                      ↓
     └────────── nuovo Forward ← Parameter Update
```

Le relazioni fondamentali tra i tre livelli sono:

```text
Autograd
  scrive i gradienti nei Tensor che richiedono gradiente
        ↓
Parameter
  è un Tensor appartenente allo stato apprendibile del modello
        ↓
Module.parameters()
  espone ricorsivamente i Parameter all'Optimizer
        ↓
Optimizer
  usa i gradienti per modificare esclusivamente i Parameter
```

## Frontiera architetturale corrente

Il ciclo di training, il broadcasting, il batch e `MatMul` per Tensor 1D/2D sono consolidati. La prossima frontiera è separare la semantica tensoriale dall'esecuzione esplicita mediante loop Python:

```text
VECTORIZATION
  ↓
BACKEND NUMERICO
  ↓
CONFRONTO CON PYTORCH
```

Questa espansione non cambia la separazione tra modello, Autograd e optimizer. Generalizza le operazioni su cui quei livelli sono costruiti.

## Mappa di lungo periodo

```text
CORE COMPUTAZIONALE
  ↓
RETI NEURALI
  ↓
TRAINING
  ↓
SCALABILITÀ
Shape → Broadcasting → Batch → MatMul generale → Vectorization
  ↓
DEEP LEARNING MODERNO
Inizializzazione → Normalizzazione → Regolarizzazione → Adam
  ↓
SEQUENZE
Embedding → Rappresentazione dei token
  ↓
ATTENTION
Q/K/V → Attention → Self-Attention → Multi-Head Attention
  ↓
TRANSFORMER
Residual → Normalization → Feed Forward → Transformer Block
  ↓
LANGUAGE MODEL
Tokenizer → Causal Mask → Next-token Prediction → Generazione
  ↓
MODELLO ESPERTO DI DOMINIO
Corpus di dominio → Adattamento / Training → Valutazione
```
