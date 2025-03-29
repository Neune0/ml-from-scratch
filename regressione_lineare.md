# Regressione lineare

## 1. Introduzione alla regressione lineare

La regressione lineare è un modello statistico che analizza la relazione tra una variabile dipendente (target) e uno o piu variabili indipendenti (feautures), stimando l'equazione di una linea che meglio approssima i dati.

Tipi di Regressione Lineare:

- Regressione lineare semplice: una sola variabile indipendente
- Regressione lineare multipla: piu variabili indipendeti

## 2. Formulazione Matematica

### Regressione Lineare Semplice

$$y = \beta_0 + \beta_1x + \varepsilon$$

Dove:

- $y$ = variabile dipendente (target)
- $x$ = variabile indipendente (feature)
- $\beta_0$ = intercetta
- $\beta_1$ = coefficiente (pendenza)
- $\varepsilon$ = termine di errore

### Regressione Lineare Multipla

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \varepsilon$$

In forma matriciale: $$\mathbf{y} = \mathbf{X\beta} + \mathbf{\varepsilon}$$

## 3. Assunzioni del modello

1. Linearità: relazione lineare tra variabili indipendenti e dipendente
2. Indipendenza: osservazioni indipendenti tra loro
3. Omoschedasticità: varianza costante degli errori
4. Normalità: distribuzione normale degli errori
5. Assenza di multicollinearità: variabili indipendenti non correlate tra loro

## 4. Stima dei Parametri (Metodo dei Minimi Quadrati)

L'obbiettivo è minimizzare la somma dei quadrati dei residui (RSS):

$$RSS = \sum_{i=1}^{n}(y_i - \hat{y}i)^2 = \sum_{i=1}^{n}(y_i - (\beta_0 + \beta_1x_i))^2$$

Le formule per stimare i coefficienti nella regressione lineare semplice sono:

$$\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

Per la regressione multipla:

 $$\mathbf{\hat{\beta}} = (\mathbf{X^TX})^{-1}\mathbf{X^Ty}$$

## 5. Valutazione del Modello

**Metriche di Performance**:

- $R^2$ (Coefficiente di determinazione): $R^2 = 1 - \frac{RSS}{TSS}$
  - Varia tra 0 e 1, dove 1 indica un fit perfetto

- MSE (Mean Square Error): $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- RMSE (Root Mean Square Error): $RMSE = \sqrt{MSE}$
- MAE (Mean Absolute Error): $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

**Diagnostica del modello**:

- Analizi dei residui
- Test di significatività (p-value)
- Intervalli di confidenza

## 6. Implementazione in Python

```py
# Esempio base con sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Generazione dati di esempio
X = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

# Inizializzazione e training del modello
model = LinearRegression()
model.fit(X, y)

# Coefficienti
print(f'Intercetta: {model.intercept_}')
print(f'Pendenza: {model.coef_[0]}')

# Predizioni
y_pred = model.predict(X)

# Valutazione
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'R²: {r2}')
print(f'RMSE: {rmse}')

# Visualizzazione
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title('Regressione Lineare Semplice')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

## 7. Problemi e soluzioni

**Overfitting**:

- Soluzione: Regolarizzazione (Ridge, Lasso)

**Multicollinearità**:

- Soluzione: Selezione delle features, PCA, Ridge Regression

**Non linearità**:

- Soluzione: Trasformazione di variabili, Regressione polinomiale

**Outliers**:

- Soluzione: regressione robusta

## 8. Varianti e estensioni

- Regressione polinomiale: $y = \beta_0 + \beta_1x + \beta_2x^2 + ... + \beta_nx^n + \varepsilon$
- Regressione Ridge: Aggiunge penalità L2 per ridurre l'overfitting
- Regressione Lasso: Aggiunge penalità L1 per selezionare le features
- Regressione Elastic Net: Combina Ridge e Lasso

## 9. Vantaggi e svantaggi

**Vantaggi**:

- Semplice da implementare e interpretare
- Computazionalmente efficiente
- Coefficienti forniscono informazioni dirette sull'impatto delle features

**Svantaggi**:

- Assume relazioni lineari
- Sensibile agli outliers
- Prestazioni limitate con relazioni complesse