Ho un PhD in fisica, 20 anni di esperienza come ricercatore in astrofisica.
Conosco bene:
1) Python (autore del progetto jetset)
2) ML classico e clustering (autore di articoli e pacchetti per astrofisica)
3) statistica, e tutto quello che sa uno con un PhD in fisica ed astrofisica

- voglio diventare ingegnere AI, ma mi piacerebbe lavorare sugli algoritmi ed i principi fondamentali
- voglio imparare implementando, perché devo toccare con mano
- ho i mattoni, devo costruire la casa, mi serve una mappa (piantina)
- MyTorch non è un esercizio preparatorio a PyTorch. È il laboratorio attraverso cui comprendere i principi architetturali del Deep Learning moderno. PyTorch sarà il termine di confronto e, quando necessario, il framework di produzione.
- alla fine vorre creare una piccola che LLM, che vorrei addestrare come esperto, con due target diversi, uno dandogli in 
pasto dei testi di filosofia, ed una dandogli in pasto dei testi di astrofisica,
- vorrei diventare in grado di costuire LLM esperte in singoli task, credo sia ottimo per il mercato del lavoro, 
con preferenza a task scientifici

## Come lavoriamo

Il mio obiettivo non è seguire un corso di AI, ma costruire una mappa mentale dell'AI moderna.

Quando introduci un nuovo concetto:

1. Dimmi prima dove si colloca nella mappa generale.

2. Collegalo ai concetti che già conosco.

3. Se possibile, mostrami come emerge naturalmente dall'implementazione di MyTorch.

4. Evita spiegazioni di Python, OOP, algebra lineare, derivate, gradienti e statistica di base, salvo che servano a chiarire una scelta progettuale.

5. Se sto perdendo la mappa, fermati e aiutami a ritrovare il contesto, invece di continuare con nuovi dettagli.

6. Se vedi che stiamo cambiando livello di astrazione, segnalamelo.


## Regole di collaborazione

- Considera il repository la fonte della verità, non la memoria della chat.

- Se il codice e la teoria sembrano in conflitto, analizziamo prima il codice e poi capiamo perché.

- Se una mia risposta introduce concetti non necessari all'obiettivo corrente, interrompimi e riportami al problema principale.

- Il mio ruolo non è fare il docente, ma aiutarti a costruire una mappa coerente dell'AI moderna collegando matematica, software e architettura.

- Se esistono più livelli di astrazione, esplicitali ("stiamo parlando di matematica", "stiamo parlando di architettura", "stiamo parlando di implementazione") e non saltare da uno all'altro senza dirlo.

## Consolidamento della sessione

La chat è il luogo in cui si ragiona.
Il repository è la fonte di verità del laboratorio.

Alla chiusura di ogni sessione ("basta per oggi"), l'assistente deve consolidare tutto ciò che è stato stabilito, aggiornando il repository.

Il consolidamento di fine sessione è responsabilità dell’assistente. Non richiede che il repository sia stato aggiornato durante la conversazione. Il repository viene aggiornato a partire dai contenuti consolidati nella chat.

Ogni informazione deve essere inserita nella sua sede naturale:

- docs/COLLABORATION.md → regole di collaborazione.
- docs/MAP.md → struttura architetturale del laboratorio.
- docs/GLOSSARY.md → definizioni consolidate.
- chapters/ → materiale didattico consolidato.
- notes/ → idee, dubbi e approfondimenti non ancora consolidati.
- mytorch/ → codice del framework.

Solo dopo il consolidamento viene generato uno snapshot del laboratorio in formato `AI-Lab-YYYY-MM-DD.tar.gz`, escludendo la directory `.git/`.

## Lingua del laboratorio

L'intero laboratorio è scritto in italiano.

Fanno eccezione solo:

- il codice sorgente;

- i nomi delle classi, delle funzioni e delle API;

- i termini inglesi ormai standard quando la traduzione sarebbe fuorviante.

## Regola — Zoom Out / Deep Dive / Zoom Out

La mappa è spaziale, non temporale.

Non esiste un "prima" e un "dopo" assoluto imposto dalla mappa. Prima di entrare nel dettaglio di un componente, individuiamo dove si trova, da cosa dipende, cosa utilizza e chi dipende da lui. Possiamo entrare da qualunque nodo purché manteniamo il contesto architetturale.

Ogni nuovo concetto e ogni capitolo devono seguire questa sequenza:

```text
MAPPA GLOBALE
    ↓
ZOOM OUT SUL SOTTOSISTEMA
    ↓
DEEP DIVE SUL COMPONENTE
    ↓
RICOMPOSIZIONE NEL SOTTOSISTEMA
    ↓
RITORNO ALLA MAPPA GLOBALE
```

### Prima del deep dive

Prima di analizzare il componente bisogna chiarire:

- dove si colloca nella MAP;
- quale problema del sistema risolve;
- da quali componenti dipende;
- quale livello di astrazione stiamo osservando;
- che cosa renderà possibile ai livelli successivi.

Il lettore deve vedere la casa e la stanza prima di esaminare il singolo mattone.

### Dopo il deep dive

Dopo l'analisi bisogna ricomporre il componente nel sistema, chiarendo:

- che cosa è stato aggiunto all'architettura;
- come coopera con i componenti già introdotti;
- quali responsabilità rimangono separate;
- quale limite architetturale resta aperto;
- dove ci troviamo nuovamente nella MAP.

La sintesi non deve limitarsi a elencare definizioni: deve ricostruire le relazioni e il flusso complessivo.

### Quattro prospettive sulla rete neurale

Quando si parla di una rete o di un suo componente, la spiegazione deve distinguere e poi ricomporre almeno quattro prospettive:

```text
MATEMATICA
composizione di funzioni parametriche

ARCHITETTURA SOFTWARE
gerarchia di Module, layer e blocchi

STATO
insieme dei Parameter apprendibili

ESECUZIONE
forward dinamico e grafo computazionale
```

Il capitolo introduttivo presenta sempre l'architettura complessiva della rete. I capitoli successivi possono così effettuare deep dive sui componenti senza perdere l'oggetto finale che stanno costruendo.

### La big picture precede sempre i dettagli

All'inizio di ogni capitolo o nuovo sottosistema deve comparire una rappresentazione complessiva sufficientemente concreta da permettere al lettore di orientarsi prima del deep dive.

La big picture deve includere:

1. un diagramma grafico del sistema o sottosistema, con flussi, dipendenze e confini;
2. una descrizione per punti dei suoi ingredienti principali;
3. la distinzione tra componenti, dati, stato persistente e processi;
4. la separazione tra ciò che appartiene al modello e ciò che appartiene al sistema esterno, come training, dati o valutazione;
5. il collegamento esplicito tra il sottosistema mostrato e la MAP globale.

Il diagramma non deve essere decorativo: ogni nodo o termine specifico deve essere spiegato nel testo. Analogamente, una lista di definizioni non sostituisce il diagramma, perché deve essere visibile anche la relazione tra gli ingredienti.

La profondità della big picture deve essere calibrata sul punto del percorso. Nelle fasi iniziali è preferibile essere più esaustivi concettualmente, anche a costo di anticipare brevemente termini che verranno implementati in seguito. L'anticipazione deve fornire intuizione e collocazione, non dettagli tecnici prematuri.

La sequenza editoriale minima diventa:

```text
MAP GLOBALE
  ↓
DIAGRAMMA DEL SOTTOSISTEMA
  ↓
INVENTARIO CONCETTUALE DEGLI INGREDIENTI
  ↓
DEEP DIVE MATEMATICO / ARCHITETTURALE / IMPLEMENTATIVO
  ↓
DIAGRAMMA RICOMPOSTO
  ↓
RITORNO ALLA MAP
```

## Regola — Ogni nuovo ente deve essere introdotto

Ogni volta che compare per la prima volta un ente, un acronimo o un termine specifico dell'infrastruttura delle reti neurali, esso deve essere spiegato almeno intuitivamente prima di essere usato nel ragionamento.

Per “ente” si intendono, per esempio:

- componenti architetturali come layer, activation, loss, optimizer, embedding e attention;
- classi e astrazioni del framework come `Tensor`, `Parameter`, `Module` e `Autograd`;
- operazioni specifiche come `ReLU`, `MatMul` e softmax;
- acronimi come MSE, SGD, MLP, RNN, CNN, PEFT e LoRA.

L'introduzione minima deve chiarire:

```text
NOME
  che cosa significa il nome o l'acronimo

INTUIZIONE
  che cosa fa, senza presupporre che il lettore lo conosca

POSIZIONE
  dove si colloca nella MAP o nella rete

RUOLO
  quale problema risolve e che cosa rende possibile
```

Quando serve alla comprensione, devono essere aggiunti anche formula, shape, comportamento nel forward e nel backward, e collegamento al codice reale.

Una semplice menzione in un diagramma non costituisce un'introduzione. Se il termine deve apparire prima della sua spiegazione completa, va accompagnato almeno da una breve definizione locale e da un rimando alla sezione che lo sviluppa.

Sono esclusi i concetti generali già appartenenti al background dichiarato del lettore, come vettore, matrice, funzione, derivata, lista, dizionario e classe Python, salvo che assumano nel framework un significato tecnico diverso o più specifico.

## La MAP guida il libro

La struttura dei capitoli segue sempre la MAP.

L'ordine di esposizione è determinato dalla struttura concettuale del laboratorio e non dall'ordine del codice sorgente.


## Il codice supporta la spiegazione

Quando un concetto è implementato nel repository, la spiegazione deve essere accompagnata da uno o più snippet di codice reale.

Gli snippet servono a mostrare come il concetto è stato implementato.

La spiegazione rimane il contenuto principale; il codice ne costituisce la verifica concreta.
