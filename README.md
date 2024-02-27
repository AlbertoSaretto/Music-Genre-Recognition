# MGR

## Update NUOVO TESTAMENTO - 09/02/2024
Cose:

* no Adagrad ma Adam ovunque: fatto. Sarà da fare esperiemnti con SGD with momentum
* aggiunta confusion matrix e f1 score. Studiare se altre metriche sono necessarie. In torchmetrics ce ne sono a bizzeffe, anche specifiche per audio
* iniziato studio trasformazioni. Vedi cartella apposita. **check on MEL Spec** sono tutti sballati. Decidere che linea seguire: io propongo training senza data augmentation e poi esperimenti con.
* Nello studio trasformazioni trovi anche plot diagnostici da eventualmente implementare

Altre cose da fare che mi sono segnato dopo l'incontro:
* capire normalizzazione. Vedi cosa fanno su `usage.ipynb`
* test set should be fixed: no random windows. Fatto.

Una volta fatte queste dovremmo essere pronti per partire con i primi esperimenti.

Altre cose:
* MixNet è da sistemare
* ResNet with finetuning could help
* LSTM layer could help
* Always check training loss plots to understand if model is underparametrized / overfit etc

----

  
## WHERE WE ARE // WHERE ARE WE GOING

**Metrics to be added. General hyperparams tuning is needed. No weights analysis done (farla alla fine)**
**remember to show explorative analysis of dataset, eg data is balanced?**

* CNN 1D: 
	- accuracy <30%
	- dropout all layers + MaxPool
	- no data augmentation
	- no PCA
	- c3 vs c4 ? (compare net with three and four layers)
	- only 6s clip -> try longer clips
	- no Optuna hyperparams tuning. Needs to be done.
	- What normalization? Try MinMax?
		- **TO DO: longer clips, data augmentation, w/wo layers, optimizers**

* CNN 2D:
	- accuracy <33%
	- MEL only (in Old trovi STFT ma è da rifare, very promising)
	- need to tune MEL BINS ( <513 bins ? )
	- 6s clips only -> try longer/shorter
	- logarithms are OK
	- data augmentation DONE
	- v2.Norm vs MinMaxScaler ? 
		-  **TO DO: many trainings with STFT only; different MELS; longer clips**

* MIXNET (1D+2D):
	- Full train (pre-trained mixed model):
		- $\sim$ 38% (validation)
	- Transfer learning: 
		- random accuracy
		- **building blocks are wrong!**

--- 
### Exotic models.
* SpatialTransformationNet: random.
* Autoencoder + ResAutoencoder : boh.
* LSTM?????



-----------------------------------------------
# MODELLI E RELATIVI PROBLEMI


-	CNN 1D: acc <30%,  optuna
-	CNN 2D : acc <40%, optuna (discutere full optuna, change lr)
-	Transfer learning: acc 12.5%  necessari più layer predictor, ma RAM. CNN non hanno buona acc di per sè quindi forse cnn non estraggono abbastanza bene le feature da essere classificate. TL puoi pompare predictor finale perché meno parametri da allenare
-	CNN 1D+2D: ? trainabile? Vanishing? Discutere numero parametri . Stare più accorti con predictor finale perché troppi parametri non stanno in RAM
-	SPATIAL TRANSFORMER NET + versione RES: spiegare cos’è e risultati training?
-	AE + ResAE: vanno trainati e va allenato predictor su spazio latente

## Problemi da spiegare: vanishing gradient.
Come è stato affrontato:
-	Inizializzato weights
-	Cambiato activation da ReLU a Sigmoid (numerical stability, gradienti che si accumulano)
-	Gradient clipping
-	Connessioni residuali + alleggerimento architettura (meno layer/neuroni)
-	Ma l’unico che ha funzionato è cambiare la normalizzazione da minmax a v2.Normalize(mean,std calcolati su batch del dataset)
**ATTENZIONE: Il problema era la mia funzione e si è risolto usando la funzione di Sklearn anziché la mia.**
## MANCA TUTTA LA PARTE DI ANALISI DEI WEIGHTS 
(vedi consegna su ppt)
