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
SCELTA DELLA STRUTTURA DEL PROBLEMA
  ├─ griglia / struttura locale ─→ CNN
  ├─ entità e relazioni ─────────→ GNN
  ├─ equazioni e vincoli fisici ─→ PINN
  └─ sequenze ───────────────────→ Embedding
                                      ↓
                                Q/K/V → Self-Attention → Multi-Head Attention
                                      ↓
                                Residual + Normalization + Feed Forward
                                      ↓
                                TRANSFORMER BLOCK
                                      ↓
                                LANGUAGE MODEL
                                Tokenizer → Causal Mask
                                  → Next-token Prediction → Generazione
```

I rami non sono alternative esclusive. Un problema scientifico può combinare più strutture, per esempio una GNN con vincoli fisici o una CNN inserita in un modello multimodale. Tutti condividono il core tensoriale, Autograd, la composizione tramite `Module` e il training loop; differiscono per bias induttivo, organizzazione dei dati e forma della loss.

```text
CNN / GNN / PINN / TRANSFORMER
  ↓
MODELLI ESPERTI DI DOMINIO
Dati scientifici o corpus di dominio
  → Training / Adattamento
  → Validazione
  → Confronto con baseline
  → Valutazione nel dominio applicativo
```

## Rami applicativi legati all'astrofisica

### CNN — struttura locale su griglie

Una **Convolutional Neural Network** condivide gli stessi kernel su regioni locali dell'input. Questo produce equivarianza rispetto alle traslazioni; pooling e head possono poi costruire una rappresentazione approssimativamente invariante quando il task lo richiede. Il bias induttivo è adatto a dati organizzati su griglie regolari.

```text
Tensor su griglia
  → Convolution
  → Feature map
  → più blocchi convoluzionali
  → Head specifica del task
```

Collegamenti astrofisici possibili:

- classificazione e segmentazione di immagini astronomiche;
- analisi di mappe del cielo;
- estrazione di feature da spettri o curve temporali con convoluzioni 1D;
- riconoscimento di strutture in output di simulazioni su griglia.

Prerequisiti principali: batch multidimensionale, operazione di convoluzione, condivisione dei pesi, padding, stride, pooling e gestione delle shape per canali e dimensioni spaziali.

### GNN — entità e relazioni

Una **Graph Neural Network** aggiorna la rappresentazione di ogni nodo aggregando messaggi provenienti dai nodi o dagli archi connessi. È adatta quando la struttura rilevante non è una griglia regolare ma un insieme di relazioni.

```text
Nodi + archi + feature
  → Message passing
  → Aggregazione locale
  → Hidden representation dei nodi o del grafo
  → Head specifica del task
```

Collegamenti astrofisici possibili:

- cataloghi di sorgenti con relazioni spaziali o fisiche;
- particelle, aloni e strutture nelle simulazioni cosmologiche;
- ricostruzione di eventi e interazioni;
- classificazione o regressione a livello di nodo, arco o intero sistema.

Prerequisiti principali: rappresentazione dei grafi, message passing, funzioni di aggregazione invarianti all'ordine, batching di grafi e readout globale.

### PINN — equazioni differenziali nella loss

Una **Physics-Informed Neural Network** rappresenta una soluzione approssimata con una rete neurale e usa Autograd per calcolarne le derivate rispetto alle coordinate fisiche. Il residuo dell'equazione e i vincoli fisici diventano termini della loss.

```text
Coordinate e parametri fisici
  → rete u_theta
  → soluzione approssimata
  → derivate tramite Autograd
  → residuo dell'equazione

LOSS TOTALE
  = loss sui dati, se disponibili
  + loss del residuo fisico
  + loss delle condizioni iniziali e al contorno
```

Prerequisiti principali: derivate rispetto agli input, derivate di ordine superiore, loss composte, sampling dei collocation points, normalizzazione delle variabili e bilanciamento dei termini della loss.

## Progetto guida PINN — Fokker–Planck nei blazar

Il progetto studierà una PINN per approssimare l'evoluzione di una distribuzione di particelle governata da un'equazione di Fokker–Planck rilevante per la modellazione dei blazar.

```text
tempo + energia + parametri fisici
  → PINN
  → distribuzione di particelle approssimata
  → residuo della Fokker–Planck
  → loss fisica + condizioni iniziali/al contorno
```

Il percorso sperimentale dovrà includere:

1. una formulazione fisica e numerica controllata del problema;
2. un solver tradizionale usato come baseline;
3. una prima PINN su un caso semplificato con soluzione nota o verificabile;
4. l'estensione a parametri fisici variabili, così che la rete possa agire come surrogate model;
5. il confronto su accuratezza, conservazione e positività quando pertinenti, stabilità e costo computazionale;
6. la distinzione tra costo iniziale di training e accelerazione delle valutazioni successive.

Il risultato atteso non è assumere che la PINN sostituisca il metodo numerico. Il progetto deve stabilire in quali regimi possa affiancarlo, accelerare scansioni parametriche o fornire una rappresentazione differenziabile della soluzione.

## Strategia MyTorch / PyTorch per i nuovi rami

```text
MYTORCH
comprensione e implementazioni minime
  → convoluzione e condivisione dei pesi
  → aggregazione e message passing
  → derivate rispetto agli input e grafi di ordine superiore

PYTORCH
esperimenti scientifici e modelli realistici
  → CNN mature
  → ecosistema per GNN
  → PINN con derivate di ordine superiore
```

Non è necessario completare in MyTorch ogni primitiva prima di usare PyTorch. Per ciascun ramo sarà identificato il nucleo architetturale da implementare in MyTorch; l'applicazione astrofisica completa verrà sviluppata con PyTorch quando ciò evita di confondere lo studio del modello con la costruzione dell'infrastruttura.
