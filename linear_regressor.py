import numpy as np
import matplotlib.pyplot as plt

# Generazione dati di esempio
np.random.seed(42)
x = np.random.rand(100) * 10
y = 2 * x + 1 + np.random.randn(100) * 2  # y = 2x + 1 + rumore

# Funzione per calcolare i coefficienti di regressione
def calcola_coefficienti(x, y):
    n = len(x)
    # Media di x e y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calcolo del coefficiente angolare (β₁)
    numeratore = np.sum((x - x_mean) * (y - y_mean))
    denominatore = np.sum((x - x_mean)**2)
    beta_1 = numeratore / denominatore
    
    # Calcolo dell'intercetta (β₀)
    beta_0 = y_mean - beta_1 * x_mean
    
    return beta_0, beta_1

# Funzione per calcolare le predizioni
def predici(x, beta_0, beta_1):
    return beta_0 + beta_1 * x

# Funzione per calcolare il coefficiente di determinazione (R²)
def calcola_r2(y, y_pred):
    # Somma dei quadrati dei residui
    ssr = np.sum((y - y_pred)**2)
    # Somma totale dei quadrati
    sst = np.sum((y - np.mean(y))**2)
    # R² = 1 - (SSR/SST)
    return 1 - (ssr / sst)

# Funzione per calcolare l'errore quadratico medio
def calcola_rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))

# Dividiamo i dati in training (80%) e test (20%)
n = len(x)
n_train = int(0.8 * n)
indices = np.random.permutation(n)
train_idx, test_idx = indices[:n_train], indices[n_train:]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

# Calcolo dei coefficienti sul set di training
beta_0, beta_1 = calcola_coefficienti(x_train, y_train)

# Previsioni sul set di training e test
y_train_pred = predici(x_train, beta_0, beta_1)
y_test_pred = predici(x_test, beta_0, beta_1)

# Calcolo delle metriche
r2_train = calcola_r2(y_train, y_train_pred)
rmse_train = calcola_rmse(y_train, y_train_pred)
r2_test = calcola_r2(y_test, y_test_pred)
rmse_test = calcola_rmse(y_test, y_test_pred)

# Visualizzazione dei risultati
plt.figure(figsize=(12, 8))

# Grafico dei dati e della retta di regressione
plt.subplot(2, 1, 1)
plt.scatter(x_train, y_train, color='blue', alpha=0.5, label='Training data')
plt.scatter(x_test, y_test, color='green', alpha=0.5, label='Test data')

# Plot della linea di regressione
x_line = np.array([min(x), max(x)])
y_line = predici(x_line, beta_0, beta_1)
plt.plot(x_line, y_line, color='red', linewidth=2, 
         label=f'y = {beta_1:.2f}x + {beta_0:.2f}')

plt.title('Regressione Lineare Semplice')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafico dei residui
plt.subplot(2, 1, 2)
plt.scatter(y_train_pred, y_train - y_train_pred, color='blue', alpha=0.5, label='Training')
plt.scatter(y_test_pred, y_test - y_test_pred, color='green', alpha=0.5, label='Test')
plt.axhline(y=0, color='red', linestyle='-', linewidth=2)
plt.title('Residui')
plt.xlabel('Valori predetti')
plt.ylabel('Residui')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Stampa dei risultati
print(f'Coefficienti del modello:')
print(f'Intercetta (β₀): {beta_0:.4f}')
print(f'Pendenza (β₁): {beta_1:.4f}')
print('\nMetriche sul training set:')
print(f'R²: {r2_train:.4f}')
print(f'RMSE: {rmse_train:.4f}')
print('\nMetriche sul test set:')
print(f'R²: {r2_test:.4f}')
print(f'RMSE: {rmse_test:.4f}')