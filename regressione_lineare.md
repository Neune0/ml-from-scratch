# Regressione lineare

## 1. Introduzione alla regressione lineare

La regressione lineare è un modello statistico che analizza la relazione tra una variabile dipendente (target) e una o più variabili indipendenti (features), stimando l'equazione di una linea che meglio approssima i dati.

Tipi di Regressione Lineare:

- Regressione lineare semplice: una sola variabile indipendente
- Regressione lineare multipla: più variabili indipendenti

## 2. Formulazione Matematica

Regressione Lineare Semplice
$$y = \beta_0 + \beta_1x + \varepsilon$$

Dove:

- $y$ = variabile dipendente (target)
- $x$ = variabile indipendente (feature)
- $\beta_0$ = intercetta
- $\beta_1$ = coefficiente (pendenza)
- $\varepsilon$ = termine di errore

Regressione Lineare Multipla
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \varepsilon$$

In forma matriciale: $$\mathbf{y} = \mathbf{X\beta} + \mathbf{\varepsilon}$$

## 3. Assunzioni del modello

- Linearità: relazione lineare tra variabili indipendenti e dipendente
- Indipendenza: osservazioni indipendenti tra loro
- Omoschedasticità: varianza costante degli errori
- Normalità: distribuzione normale degli errori
- Assenza di multicollinearità: variabili indipendenti non correlate tra loro

## 4. Funzioni di Costo per la Regressione
Le funzioni di costo quantificano la discrepanza tra i valori predetti e quelli reali, guidando l'ottimizzazione dei parametri del modello.

### 4.1 Funzioni di Base

- Mean Squared Error (MSE):
  - $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
  - La funzione di costo più comune per la regressione lineare
  - Penalizza maggiormente errori grandi rispetto a quelli piccoli
- Root Mean Squared Error (RMSE)
  - $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
  - Stessa unità di misura della variabile target
  - Facilita l'interpretazione dell'errore
- Mean Absolute Error (MAE) 
  - $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
  - Più robusto agli outlier rispetto a MSE
  - Interpreta l'errore medio in valore assoluto
- Sum of Squared Errors (SSE) / Residual Sum of Squares (RSS) $$\text{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
  - Utilizzata nella derivazione dei parametri OLS
  - Non normalizzata per il numero di campioni
- R-squared (Coefficiente di determinazione) 
  - $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}i)^2}{\sum{i=1}^{n}(y_i - \bar{y})^2} = 1 - \frac{\text{SSE}}{\text{TSS}}$$
  - Rappresenta la proporzione della varianza spiegata dal modello
  - Varia tra 0 e 1 (o negativo in casi patologici)
- Adjusted R-squared 
  - $$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$
  - Corregge $R^2$ considerando il numero di predittori $p$
  - Penalizza modelli con troppe variabili

### 4.2 Funzioni Robuste e Specializzate

- Huber Loss 
  - $$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{per } |y - \hat{y}| \leq \delta \ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{altrimenti} \end{cases}$$
  - Combina MSE per errori piccoli e MAE per errori grandi
  - Più robusta agli outlier rispetto a MSE
- Mean Absolute Percentage Error (MAPE) 
  - $$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
  - Esprime l'errore come percentuale del valore reale
  - Problematico quando $y_i$ è vicino a zero
- Symmetric Mean Absolute Percentage Error (SMAPE) 
  - $$\text{SMAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$
  - Versione simmetrica e limitata del MAPE
  - Varia tra 0% e 200%
- Quantile Loss / Pinball Loss
  - $$L_\tau(y, \hat{y}) = \begin{cases} \tau(y - \hat{y}), & \text{se } y - \hat{y} \geq 0 \ (1-\tau)(|\hat{y} - y|), & \text{se } y - \hat{y} < 0 \end{cases}$$
  - Usata per la regressione quantilica ($\tau$ è il quantile desiderato)
  - Permette di modellare diversi percentili della distribuzione
- Log-Cosh Loss
  - $$L(y, \hat{y}) = \sum_{i=1}^{n} \log(\cosh(y_i - \hat{y}_i))$$
  - Approssima MSE per errori piccoli e MAE per errori grandi
  - Due volte differenziabile ovunque

### 4.3 Funzioni di Costo Regolarizzate

Ridge Regression (L2) $$L_{\text{Ridge}} = \text{MSE} + \lambda \sum_{j=1}^{p} \beta_j^2$$

Aggiunge penalità proporzionale al quadrato dei coefficienti
Riduce l'overfitting contraendo i coefficienti
Lasso Regression (L1) $$L_{\text{Lasso}} = \text{MSE} + \lambda \sum_{j=1}^{p} |\beta_j|$$

Promuove sparsità dei coefficienti
Può portare a zero i coefficienti meno importanti
Elastic Net $$L_{\text{ElasticNet}} = \text{MSE} + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2$$

Combina regolarizzazione L1 e L2
Utile per dataset con variabili altamente correlate
4.4 Scelta della Funzione di Costo
La scelta dipende da vari fattori:

Tipo di dati: La distribuzione e le caratteristiche dei dati
Sensibilità agli outlier: Alcune metriche sono più robuste di altre
Interpretabilità: Metriche come RMSE sono più facili da interpretare
Obiettivo specifico: Per esempio, la necessità di modellare quantili o percentuali
Dominio applicativo: Diverse aree possono avere convenzioni o requisiti specifici
5. Derivate e Discesa del Gradiente
5.1 Derivate: Un'Intuizione Geometrica
La derivata di una funzione in un punto rappresenta la pendenza della tangente alla curva in quel punto. È un concetto fondamentale che ci permette di comprendere come una funzione varia localmente.

Interpretazione geometrica:
Derivata positiva: La funzione cresce (pendenza verso l'alto)
Derivata negativa: La funzione decresce (pendenza verso il basso)
Derivata zero: Punto stazionario (massimo, minimo o punto di flesso)
<img alt="Interpretazione geometrica della derivata" src="https://i.imgur.com/xgJnPR1.png">
Significato del valore della derivata:
Il valore numerico della derivata in un punto rappresenta:

Inclinazione della tangente: Maggiore è il valore assoluto, più ripida è la curva
Velocità di variazione: Quanto rapidamente la funzione cambia in quel punto
Direzione di variazione: Positiva (crescente) o negativa (decrescente)
Per esempio, una derivata di valore 2 indica che la funzione cresce di 2 unità sull'asse y per ogni incremento di 1 unità sull'asse x.

Approssimazione lineare tramite derivata:
La derivata ci permette di approssimare localmente la funzione con una retta tangente:

$$f(x+h) \approx f(x) + f'(x) \cdot h$$

Questa approssimazione migliora man mano che $h$ diventa più piccolo, ed è alla base del metodo della discesa del gradiente.

Definizione matematica:
La derivata di una funzione $f(x)$ nel punto $x$ è definita come:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

Interpretazione nella regressione lineare:
Nella regressione lineare, la derivata della funzione di costo rispetto a un parametro indica:

Direzione di variazione: Come cambierebbe il costo incrementando o decrementando leggermente il parametro
Sensibilità dell'errore: Quali parametri hanno maggiore impatto sull'errore complessivo
Direzione di aggiornamento: Come modificare i parametri per ridurre l'errore
Quando la derivata è zero, abbiamo raggiunto un punto stazionario. Nel caso della funzione di costo della regressione lineare (una funzione convessa), questo punto rappresenta l'unico minimo globale, cioè i parametri ottimali che minimizzano l'errore.

<img alt="Derivata come pendenza" src="https://i.imgur.com/QkXMbjZ.png">
Le derivate possono essere visualizzate anche come la pendenza di una superficie nello spazio 3D, dove più la superficie è inclinata, maggiore è il valore della derivata in quella direzione.

5.2 Derivate Parziali
Quando una funzione dipende da più variabili, come nel caso della regressione lineare multipla, utilizziamo le derivate parziali. La derivata parziale di $f(x,y)$ rispetto a $x$ (indicata con $\frac{\partial f}{\partial x}$) esprime come varia la funzione quando cambia solo $x$, mantenendo $y$ costante.

Per una funzione di costo $J(\beta_0, \beta_1)$ nella regressione lineare semplice:

$\frac{\partial J}{\partial \beta_0}$ indica come varia il costo al variare dell'intercetta
$\frac{\partial J}{\partial \beta_1}$ indica come varia il costo al variare della pendenza
5.3 Gradiente
Il gradiente di una funzione è un vettore che contiene tutte le derivate parziali. Geometricamente, il gradiente punta nella direzione di massima crescita della funzione.

Per la funzione di costo $J(\beta_0, \beta_1)$, il gradiente è:

$$\nabla J = \begin{bmatrix} \frac{\partial J}{\partial \beta_0} \ \frac{\partial J}{\partial \beta_1} \end{bmatrix}$$

5.4 Discesa del Gradiente: Intuizione Grafica
Immaginiamo di trovarci su una superficie tridimensionale che rappresenta la funzione di costo, dove l'altezza in ogni punto $(\beta_0, \beta_1)$ indica il valore della funzione di costo per quei parametri.

<img alt="Discesa del gradiente in una superficie 3D" src="https://i.imgur.com/JYOWDvg.png">
La discesa del gradiente funziona come segue:

Iniziamo in un punto arbitrario sulla superficie
Calcoliamo il gradiente in quel punto (che indica la direzione di massima crescita)
Ci muoviamo nella direzione opposta al gradiente (verso il basso)
Ripetiamo finché non raggiungiamo un minimo (o fino a convergenza)
5.5 Esempio: Trovare il Minimo di una Parabola
Consideriamo la semplice funzione $f(x) = x^2$, che ha il suo minimo in $x=0$. La derivata di questa funzione è $f'(x) = 2x$.

Se applichiamo la discesa del gradiente partendo da un punto $x_0$, l'aggiornamento a ogni iterazione sarà:

$$x_{t+1} = x_t - \alpha \cdot f'(x_t) = x_t - \alpha \cdot 2x_t = (1 - 2\alpha)x_t$$

Dove $\alpha$ è il learning rate (tasso di apprendimento).

Se partiamo da $x_0 = 5$ e usiamo $\alpha = 0.1$:

$x_1 = 5 - 0.1 \cdot 2 \cdot 5 = 5 - 1 = 4$
$x_2 = 4 - 0.1 \cdot 2 \cdot 4 = 4 - 0.8 = 3.2$
$x_3 = 3.2 - 0.1 \cdot 2 \cdot 3.2 = 3.2 - 0.64 = 2.56$
...
Ad ogni iterazione ci avviciniamo sempre più al minimo della parabola in $x=0$.

<img alt="Discesa del gradiente su una parabola" src="https://i.imgur.com/9XfNY5l.png">
5.6 Derivazione del Gradiente per Diverse Funzioni di Costo
Per ogni funzione di costo nella regressione lineare semplice ($\hat{y}_i = \beta_0 + \beta_1 x_i$), deriviamo il gradiente necessario per la discesa del gradiente:

1. Mean Squared Error (MSE)
Funzione di costo: $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}i)^2 = \frac{1}{n}\sum{i=1}^{n}(y_i - (\beta_0 + \beta_1 x_i))^2$$

Derivate parziali: $$\frac{\partial \text{MSE}}{\partial \beta_0} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)$$

$$\frac{\partial \text{MSE}}{\partial \beta_1} = -\frac{2}{n}\sum_{i=1}^{n}x_i(y_i - \beta_0 - \beta_1 x_i)$$

2. Root Mean Squared Error (RMSE)
Funzione di costo: $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - (\beta_0 + \beta_1 x_i))^2}$$

Derivate parziali (usando regola della catena): $$\frac{\partial \text{RMSE}}{\partial \beta_0} = -\frac{1}{n \cdot \text{RMSE}}\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)$$

$$\frac{\partial \text{RMSE}}{\partial \beta_1} = -\frac{1}{n \cdot \text{RMSE}}\sum_{i=1}^{n}x_i(y_i - \beta_0 - \beta_1 x_i)$$

3. Mean Absolute Error (MAE)
Funzione di costo: $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - (\beta_0 + \beta_1 x_i)|$$

Derivate parziali (usando la funzione signum): $$\frac{\partial \text{MAE}}{\partial \beta_0} = -\frac{1}{n}\sum_{i=1}^{n}\text{sgn}(y_i - \beta_0 - \beta_1 x_i)$$

$$\frac{\partial \text{MAE}}{\partial \beta_1} = -\frac{1}{n}\sum_{i=1}^{n}x_i \cdot \text{sgn}(y_i - \beta_0 - \beta_1 x_i)$$

5.7 Scelta del Learning Rate
La scelta del learning rate $\alpha$ è cruciale:

Troppo piccolo: la convergenza sarà lenta
Troppo grande: si rischia di "saltare" oltre il minimo e divergere
Nella pratica, si possono utilizzare tecniche come il learning rate adattivo o la ricerca lineare per ottimizzare questo parametro.

6. Stima dei Parametri
Esistono due approcci principali per stimare i parametri della regressione lineare:

6.1 Metodo dei Minimi Quadrati (Soluzione Analitica)
Per la regressione lineare semplice $y = \beta_0 + \beta_1x + \varepsilon$, l'obiettivo è minimizzare la somma dei quadrati dei residui (RSS):

$$RSS = \sum_{i=1}^{n}(y_i - \hat{y}i)^2 = \sum{i=1}^{n}(y_i - (\beta_0 + \beta_1x_i))^2$$

Impostando le derivate parziali rispetto a $\beta_0$ e $\beta_1$ uguali a zero, otteniamo:

$$\frac{\partial RSS}{\partial \beta_0} = -2\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1x_i) = 0$$

$$\frac{\partial RSS}{\partial \beta_1} = -2\sum_{i=1}^{n}x_i(y_i - \beta_0 - \beta_1x_i) = 0$$

Risolvendo questo sistema di equazioni otteniamo le formule chiuse:

$$\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

Per la regressione multipla, la soluzione in forma matriciale è:

$$\mathbf{\hat{\beta}} = (\mathbf{X^TX})^{-1}\mathbf{X^Ty}$$

6.2 Discesa del Gradiente (Soluzione Iterativa)
Quando la dimensione dei dati è elevata o la matrice $\mathbf{X^TX}$ non è facilmente invertibile, possiamo utilizzare la discesa del gradiente. Partendo da valori iniziali arbitrari, aggiorniamo iterativamente i parametri nella direzione opposta al gradiente della funzione di costo.

Varianti della Discesa del Gradiente
Batch Gradient Descent: Utilizza tutti i dati per calcolare il gradiente ad ogni iterazione.
Stochastic Gradient Descent (SGD): Aggiorna i parametri usando un solo campione alla volta.
Formula: $$\beta_0^{(t+1)} = \beta_0^{(t)} + \alpha \cdot 2e_i$$ $$\beta_1^{(t+1)} = \beta_1^{(t)} + \alpha \cdot 2x_i \cdot e_i$$
Mini-Batch Gradient Descent: Compromesso che usa un sottoinsieme di dati per iterazione.
7. Valutazione del Modello
Metriche di Performance:

$R^2$ (Coefficiente di determinazione): $R^2 = 1 - \frac{RSS}{TSS}$

Varia tra 0 e 1, dove 1 indica un fit perfetto
MSE (Mean Square Error): $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

RMSE (Root Mean Square Error): $RMSE = \sqrt{MSE}$

MAE (Mean Absolute Error): $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

Diagnostica del modello:

Analisi dei residui
Test di significatività (p-value)
Intervalli di confidenza
8. Implementazione in Python
9. Problemi e soluzioni
Overfitting:

Soluzione: Regolarizzazione (Ridge, Lasso)
Multicollinearità:

Soluzione: Selezione delle features, PCA, Ridge Regression
Non linearità:

Soluzione: Trasformazione di variabili, Regressione polinomiale
Outliers:

Soluzione: Regressione robusta
10. Varianti e estensioni
Regressione polinomiale: $y = \beta_0 + \beta_1x + \beta_2x^2 + ... + \beta_nx^n + \varepsilon$
Regressione Ridge: Aggiunge penalità L2 per ridurre l'overfitting
Regressione Lasso: Aggiunge penalità L1 per selezionare le features
Regressione Elastic Net: Combina Ridge e Lasso
11. Vantaggi e svantaggi
Vantaggi:

Semplice da implementare e interpretare
Computazionalmente efficiente
Coefficienti forniscono informazioni dirette sull'impatto delle features
Svantaggi:

Assume relazioni lineari
Sensibile agli outliers
Prestazioni limitate con relazioni complesse