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

## Lingua 
 quando scrivi nel repo, eccetto COLLABORATION.md, tutta va in Inglese