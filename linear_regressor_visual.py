import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generazione dati di esempio (riutilizzati dal file originale)
np.random.seed(42)
x = np.random.rand(100) * 10
y = 2 * x + 1 + np.random.randn(100) * 2  # y = 2x + 1 + rumore

# Normalizzazione delle feature per velocizzare la convergenza
x_norm = (x - np.mean(x)) / np.std(x)

# Apprendimento con discesa del gradiente
def gradient_descent(x, y, learning_rate=0.01, num_iterations=100):
    # Inizializzazione dei parametri
    m = len(y)
    theta0 = 0  # intercetta
    theta1 = 0  # pendenza
    
    # Storia dei parametri e costi per visualizzazione
    theta_history = np.zeros((num_iterations, 2))
    cost_history = np.zeros(num_iterations)
    
    # Discesa del gradiente
    for i in range(num_iterations):
        # Previsioni con parametri correnti
        y_pred = theta0 + theta1 * x
        
        # Calcolo dei gradienti
        gradient_theta0 = (1/m) * np.sum(y_pred - y)
        gradient_theta1 = (1/m) * np.sum((y_pred - y) * x)
        
        # Aggiornamento parametri
        theta0 = theta0 - learning_rate * gradient_theta0
        theta1 = theta1 - learning_rate * gradient_theta1
        
        # Salvataggio storia
        theta_history[i] = [theta0, theta1]
        
        # Calcolo errore quadratico medio
        cost_history[i] = (1/(2*m)) * np.sum((y_pred - y)**2)
    
    return theta0, theta1, theta_history, cost_history

# Esecuzione della discesa del gradiente
iterations = 100
theta0, theta1, theta_history, cost_history = gradient_descent(x_norm, y, 
                                                               learning_rate=0.1, 
                                                               num_iterations=iterations)

# Denormalizzazione dei parametri
x_mean = np.mean(x)
x_std = np.std(x)
beta1 = theta1 / x_std
beta0 = theta0 - beta1 * x_mean

# Creazione della figura per l'animazione
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
fig.tight_layout(pad=5)

# Preparazione del grafico 1: dati e linea di regressione
ax1.scatter(x, y, color='blue', alpha=0.5, label='Dati')
line, = ax1.plot([], [], 'r-', linewidth=2, label='Retta di regressione')
ax1.set_xlim(0, 11)
ax1.set_ylim(min(y)-2, max(y)+2)
ax1.set_title('Evoluzione della Regressione Lineare')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Testo per visualizzare l'equazione corrente
equation_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, 
                         fontsize=10, verticalalignment='top')

# Preparazione del grafico 2: funzione di costo
ax2.set_xlim(0, iterations)
ax2.set_ylim(0, cost_history[0] * 1.1)
ax2.set_title('Funzione di Costo (MSE)')
ax2.set_xlabel('Iterazioni')
ax2.set_ylabel('Errore Quadratico Medio')
ax2.grid(True, alpha=0.3)
cost_line, = ax2.plot([], [], 'b-')

# Funzione per l'inizializzazione dell'animazione
def init():
    line.set_data([], [])
    cost_line.set_data([], [])
    equation_text.set_text('')
    return line, cost_line, equation_text

# Funzione per aggiornare ogni frame dell'animazione
def update(frame):
    # Denormalizzazione dei parametri per questo frame
    b1 = theta_history[frame, 1] / x_std
    b0 = theta_history[frame, 0] - b1 * x_mean
    
    # Aggiornamento della linea di regressione
    x_line = np.array([0, 10])
    y_line = b0 + b1 * x_line
    line.set_data(x_line, y_line)
    
    # Aggiornamento del testo dell'equazione
    equation_text.set_text(f'y = {b1:.4f}x + {b0:.4f}\nIterazione: {frame}/{iterations}')
    
    # Aggiornamento del grafico della funzione di costo
    iterations_so_far = np.arange(frame + 1)
    cost_line.set_data(iterations_so_far, cost_history[:frame + 1])
    
    return line, cost_line, equation_text

# Creazione dell'animazione
ani = FuncAnimation(fig, update, frames=iterations,
                    init_func=init, blit=True, interval=50, repeat=False)

plt.show()

# Opzionale: salva l'animazione come file
ani.save('regressione_lineare.gif', writer='pillow', fps=20)

print(f'Coefficienti finali del modello:')
print(f'Intercetta (β₀): {beta0:.4f}')
print(f'Pendenza (β₁): {beta1:.4f}')