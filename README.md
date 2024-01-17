# MGR 
## WHERE WE ARE // WHERE ARE WE GOING

** Metrics to be added. General hyperparams tuning is needed. **

* CNN 1D: 
	- accuracy <30%
	- dropout all layers + MaxPool
	- no data augmentation
	- no PCA
	- c3 vs c4 ? (compare net with three and four layers)
	- only 6s clip -> try longer clips
	- no Optuna hyperparams tuning. Needs to be done.
	- What normalization? Try MinMax?
        	- ** TO DO: longer clips, data augmentation, w/wo layers, optimizers**

* CNN 2D:
	- accuracy <33%
	- MEL only (in Old trovi STFT ma è da rifare, very promising)
	- need to tune MEL BINS ( <513 bins ? )
	- 6s clips only -> try longer/shorter
	- logarithms are OK
	- data augmentation DONE
	- v2.Norm vs MinMaxScaler ? 
		-  **TO DO: many trainings with STFT only; different MELS; longer clips **

* MIXNET (1D+2D):
	- Full train (pre-trained mixed model):
		- $\sim$ 38% 




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
**ATTENZIONE: Il problema era la mia funzione e si è risolto usando la funzione di Sklearn anziché la mia. Io mentirei e direi che si è risolto usando queste tecniche...**

## MANCA TUTTA LA PARTE DI ANALISI DEI WEIGHTS 
(vedi consegna su ppt)
