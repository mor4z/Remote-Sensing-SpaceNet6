# Remote-Sensing-SpaceNet6

Repository per il progetto del corso "Laboratorio di Intelligenza Artificiale" tenuto dal prof. Ciarfuglia dell'università La Sapienza di Roma

## Procedura di lavoro da seguire

### 1. Preparazione del dataset
- Preparare le maschere che saranno le ground truth dell'addestramento. Convertire le label contenute nei file .geojson in maschere binarie in formato .tiff, come suggerito dal tutor.
- Creare una classe dataset per caricare i dati
- Suddividere il dataset in training, validation e test set (80%, 10%, 10%).
- Salvare i percorsi delle immagini e delle label in 3 diversi file .txt per tenerne traccia. Vanno usate sempre le stesse suddivisioni.

Suggerimento del tutor: ci sono dei file di troppo, utilizza solo i file goejson che hanno una corrispondenza 1 a 1 con i file tiff delle immagini e le label geojson. Le label andranno convertite da geojson a rasterio per utilizzare come ground truth (per questa operazione si può usare rasterio o solaris).

### 2. Costruzione della rete neurale convoluzionale
- Costruire un modello UNET

Suggerimento del tutor: prima del training settare il seed per migliorare la riproducibilità.

### 3. Addestramento della rete neurale
- Scegliere la funzione di Loss
I training che da eseguire sono almeno i seguenti:
- Baseline (prova diversi learning rate su scala logaritmica, e.g. 1e-2, 1e-3, 1e-4, 1e-5, una volta individuato il migliore passa al punto successivo).
- Aggiunta di data augmentation (per esempio random horizontal/vertical flip, rotazioni di 0/90/180/270 gradi). Come libreria usare Albumentations.

Durante il training salvare i seguenti dati per ogni epoca:
- train loss
- validation loss
- precision
- recall
- F1-score
- accuracy sul validation set

Ricorda di tracciare i grafici delle loss e degli score per ogni training. Al termine dell'addestramento salvare il modello con F1-Score migliore.

### 4. Valutazione del modello migliore sul test set
Verificare l'efficienza del modello migliore sul test set, facendo una tabella che contenga le seguenti metriche:
- precision
- recall
- F1-score
- accuracy

Realizzare infine la matrice di confusione.

Nel corso del progetto salvare tutti i dati di training/pesi/grafici che devono poi essere mostrati all'esame.