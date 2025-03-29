import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class NewtonPolynomialRegression:
    """
    Implementazione di regressione polinomiale usando i polinomi di Newton.
    
    I polinomi di Newton hanno la forma:
    p(x) = c₀ + c₁(x-x₀) + c₂(x-x₀)(x-x₁) + ... + cₙ(x-x₀)(x-x₁)...(x-xₙ₋₁)
    
    Questa implementazione usa una regressione dei minimi quadrati per trovare
    i coefficienti ottimali, piuttosto che l'interpolazione esatta.
    """
    
    def __init__(self, degree: int = 3):
        """
        Inizializza il modello di regressione polinomiale.
        
        Args:
            degree: Grado del polinomio da utilizzare
        """
        self.degree = degree
        self.coefficients = None
        self.nodes = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'NewtonPolynomialRegression':
        """
        Addestra il modello sui dati usando i polinomi di Newton.
        
        Args:
            x: Feature di input
            y: Target values
            
        Returns:
            Il modello addestrato (self)
        """
        # Converti input in array numpy
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        # Usa punti equidistanti come nodi per il polinomio di Newton
        self.nodes = np.linspace(np.min(x), np.max(x), self.degree)
        
        # Costruisci la matrice di Vandermonde modificata per base di Newton
        V = self._build_newton_basis(x)
        
        # Calcola i coefficienti usando minimi quadrati
        self.coefficients, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        
        return self
    
    def _build_newton_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Costruisce la matrice della base di Newton per i dati di input.
        
        Args:
            x: Punti dove valutare la base
            
        Returns:
            Matrice della base di Newton
        """
        n = len(x)
        V = np.ones((n, self.degree))
        
        # Implementazione corretta che evita problemi di broadcasting
        for i in range(n):
            # Per ogni punto x[i]
            for j in range(1, self.degree):
                # Calcola il prodotto (x[i]-nodes[0])*(x[i]-nodes[1])*...(x[i]-nodes[j-1])
                term = 1.0
                for k in range(j):
                    term *= (x[i] - self.nodes[k])
                V[i, j] = term
            
        return V
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Effettua previsioni sui dati di input.
        
        Args:
            x: Dati di input
            
        Returns:
            Previsioni
        """
        if self.coefficients is None or self.nodes is None:
            raise ValueError("Il modello non è stato addestrato. Chiamare fit() prima di predict().")
            
        x = np.asarray(x).flatten()
        V = self._build_newton_basis(x)
        
        return V @ self.coefficients
    
    def get_polynomial_str(self) -> str:
        """
        Restituisce una rappresentazione in stringa del polinomio.
        
        Returns:
            Stringa che rappresenta il polinomio
        """
        if self.coefficients is None or self.nodes is None:
            raise ValueError("Il modello non è stato addestrato.")
            
        terms = [f"{self.coefficients[0]:.4f}"]
        
        for i in range(1, self.degree):
            # Costruisci il termine del prodotto
            product_terms = []
            for j in range(i):
                product_terms.append(f"(x-{self.nodes[j]:.4f})")
                
            product_str = " * ".join(product_terms)
            terms.append(f"{self.coefficients[i]:.4f} * {product_str}")
            
        return " + ".join(terms)

class ModelEvaluator:
    """Classe per la valutazione dei modelli di regressione"""
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcola il coefficiente di determinazione (R²)
        
        Args:
            y_true: Valori reali
            y_pred: Valori predetti
            
        Returns:
            Coefficiente R²
        """
        # Somma dei quadrati dei residui
        ssr = np.sum((y_true - y_pred)**2)
        # Somma totale dei quadrati
        sst = np.sum((y_true - np.mean(y_true))**2)
        
        if sst == 0:
            return 0.0  # Per evitare divisione per zero
            
        # R² = 1 - (SSR/SST)
        return 1 - (ssr / sst)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcola l'errore quadratico medio
        
        Args:
            y_true: Valori reali
            y_pred: Valori predetti
            
        Returns:
            RMSE
        """
        return np.sqrt(np.mean((y_true - y_pred)**2))

class Visualizer:
    """Classe per la visualizzazione dei risultati"""
    
    @staticmethod
    def plot_polynomial_regression(x, y, model, title="Regressione Polinomiale con Polinomi di Newton"):
        """
        Visualizza il grafico della regressione polinomiale e dei residui
        
        Args:
            x: Features
            y: Target values
            model: Modello addestrato
            title: Titolo del grafico
        """
        # Crea punti per una curva liscia
        x_line = np.linspace(min(x), max(x), 1000)
        y_line = model.predict(x_line)
        
        # Previsioni sui dati originali
        y_pred = model.predict(x)
        
        # Calcolo metriche
        r2 = ModelEvaluator.r2_score(y, y_pred)
        rmse = ModelEvaluator.rmse(y, y_pred)
        
        # Crea il grafico
        plt.figure(figsize=(12, 10))
        
        # Grafico dati e curva
        plt.subplot(2, 1, 1)
        plt.scatter(x, y, color='blue', alpha=0.6, label='Dati')
        plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Polinomio di grado {model.degree-1}')
        
        plt.title(f'{title}\nGrado: {model.degree-1}, R²: {r2:.4f}, RMSE: {rmse:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Grafico dei residui
        plt.subplot(2, 1, 2)
        plt.scatter(y_pred, y - y_pred, color='blue', alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='-', linewidth=2)
        plt.title('Residui')
        plt.xlabel('Valori predetti')
        plt.ylabel('Residui')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Stampa dei risultati
        print(f'Metriche del modello:')
        print(f'R²: {r2:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'\nForma polinomiale di Newton:')
        try:
            print(model.get_polynomial_str())
        except:
            print("Non è stato possibile generare la rappresentazione in stringa del polinomio.")

def generate_sample_data(size=100, noise=0.5, random_state=None):
    """
    Genera dati di esempio per regressione polinomiale
    
    Args:
        size: Numero di campioni
        noise: Deviazione standard del rumore
        random_state: Seed per riproducibilità
        
    Returns:
        x, y: Feature e target
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    x = np.linspace(-3, 3, size)
    # Funzione non lineare: y = x³ - 2x² + 2 + rumore
    y = x**3 - 2*x**2 + 2 + np.random.randn(size) * noise
    
    return x, y

# Esempio di utilizzo
if __name__ == "__main__":
    # Generiamo dati di esempio non lineari
    x, y = generate_sample_data(size=100, noise=1.0, random_state=42)
    
    # Creiamo e addestriamo modelli di diverso grado
    degrees = [2, 4, 10, 15, 20, 25, 30, 35,40]
    
    for degree in degrees:
        # Il grado effettivo è degree - 1 perché includiamo il termine costante
        model = NewtonPolynomialRegression(degree=degree)
        model.fit(x, y)
        
        # Visualizziamo i risultati
        Visualizer.plot_polynomial_regression(x, y, model, 
                                             title=f"Regressione Polinomiale (Newton) di Grado {degree-1}")
    
    # Mostriamo un esempio di overfitting con grado alto
    np.random.seed(42)
    # Usiamo meno punti per evidenziare l'overfitting
    x_sparse, y_sparse = generate_sample_data(size=20, noise=1.0, random_state=42)
    
    model_high = NewtonPolynomialRegression(degree=15)
    model_high.fit(x_sparse, y_sparse)
    
    Visualizer.plot_polynomial_regression(x_sparse, y_sparse, model_high, 
                                         title="Esempio di Overfitting con Polinomio di Grado Alto")