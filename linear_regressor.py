import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Tuple, Optional, Union

class FittingMethod(Enum):
    GRADIENT_DESCENT = "gradient_descent"
    CLOSED_FORM = "closed_form"

class Dataset:
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.array(x)
        self.y = np.array(y)
        
        # Controllo dimensioni
        if len(self.x) != len(self.y):
            raise ValueError("x e y devono avere la stessa lunghezza")
    
    def normalize(self) -> None:
        """Normalizza le features tra 0 e 1"""
        self.x = (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
    
    def standardize(self) -> None:
        """Standardizza le features (media 0, deviazione standard 1)"""
        self.x = (self.x - np.mean(self.x)) / np.std(self.x)
    
    def split(self, train_size: float = 0.8, random_state: Optional[int] = None) -> Tuple:
        if random_state is not None:
            np.random.seed(random_state)
            
        n = len(self.x)
        n_train = int(train_size * n)
        
        # Genera indici casuali
        indices = np.random.permutation(n)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        # Divide in training e test set
        x_train, y_train = self.x[train_idx], self.y[train_idx]
        x_test, y_test = self.x[test_idx], self.y[test_idx]
        
        return (x_train, y_train), (x_test, y_test)
    
    @classmethod
    def generate_sample_data(cls, size=100, slope=2, intercept=1, noise=2, random_state=None):
        """
        Genera un nuovo dataset di esempio per regressione lineare
        
        Args:
            size: Numero di campioni
            slope: Pendenza reale
            intercept: Intercetta reale
            noise: Deviazione standard del rumore
            random_state: Seed per riproducibilità
            
        Returns:
            Dataset: una nuova istanza di Dataset con i dati generati
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        x = np.random.rand(size) * 10
        y = slope * x + intercept + np.random.randn(size) * noise
        
        return cls(x, y)  # Restituisce una nuova istanza di Dataset

class LinearRegressionModel:
    """Modello di regressione lineare y = β₀ + β₁x"""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        """
        Inizializza il modello di regressione lineare
        
        Args:
            learning_rate: Tasso di apprendimento per gradient descent
            num_iterations: Numero di iterazioni per gradient descent
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta_0 = None  # Intercetta
        self.beta_1 = None  # Pendenza
        self.history = []   # Storico dell'errore durante il training
    
    def fit(self, x: np.ndarray, y: np.ndarray, method: FittingMethod = FittingMethod.CLOSED_FORM) -> 'LinearRegressionModel':
        """
        Addestra il modello sui dati
        
        Args:
            x: Feature di input
            y: Target values
            method: Metodo di fitting (gradient_descent o closed_form)
            
        Returns:
            Il modello addestrato (self)
        """
        if method == FittingMethod.GRADIENT_DESCENT:
            return self._fit_gradient_descent(x, y)
        else:
            return self._fit_closed_form(x, y)
    
    def _fit_gradient_descent(self, x: np.ndarray, y: np.ndarray) -> 'LinearRegressionModel':
        """Addestra il modello usando gradient descent"""
        n = len(x)
        self.beta_0 = 0.0  # Inizializzazione dell'intercetta
        self.beta_1 = 0.0  # Inizializzazione della pendenza
        self.history = []  # Reset dello storico
        
        for _ in range(self.num_iterations):
            y_pred = self.predict(x)
            error = y_pred - y
            mse = np.mean(error**2)
            self.history.append(mse)
            
            # Aggiornamento dei coefficienti
            self.beta_0 -= (self.learning_rate / n) * np.sum(error)
            self.beta_1 -= (self.learning_rate / n) * np.sum(error * x)
        
        return self
    
    def _fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> 'LinearRegressionModel':
        """Addestra il modello usando la formula chiusa"""
        # Media di x e y
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calcolo del coefficiente angolare (β₁)
        numeratore = np.sum((x - x_mean) * (y - y_mean))
        denominatore = np.sum((x - x_mean)**2)
        
        if denominatore == 0:
            raise ValueError("Impossibile calcolare la pendenza: varianza di x è zero")
            
        self.beta_1 = numeratore / denominatore
        
        # Calcolo dell'intercetta (β₀)
        self.beta_0 = y_mean - self.beta_1 * x_mean
        
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Effettua previsioni per i dati di input
        
        Args:
            x: Features di input
            
        Returns:
            Previsioni
        """
        if self.beta_0 is None or self.beta_1 is None:
            raise ValueError("Il modello non è stato addestrato. Chiamare fit() prima di predict().")
            
        return self.beta_0 + self.beta_1 * x
    
    def get_params(self) -> dict:
        """
        Restituisce i parametri del modello
        
        Returns:
            Dizionario con i parametri
        """
        return {"intercetta": self.beta_0, "pendenza": self.beta_1}
    
    def plot_training_history(self) -> None:
        """Visualizza lo storico dell'errore durante il training"""
        if not self.history:
            print("Nessun dato di storia disponibile. Usa il metodo gradient_descent per il training.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.history)
        plt.title('Andamento dell\'errore (MSE) durante il training')
        plt.xlabel('Iterazioni')
        plt.ylabel('Errore quadratico medio')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

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
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcola l'errore assoluto medio
        
        Args:
            y_true: Valori reali
            y_pred: Valori predetti
            
        Returns:
            MAE
        """
        return np.mean(np.abs(y_true - y_pred))

class Visualizer:
    """Classe per la visualizzazione dei risultati"""
    
    @staticmethod
    def plot_regression(x_train, y_train, x_test, y_test, model, title="Regressione Lineare"):
        """
        Visualizza il grafico della regressione e dei residui
        
        Args:
            x_train: Features di training
            y_train: Target di training
            x_test: Features di test
            y_test: Target di test
            model: Modello addestrato
            title: Titolo del grafico
        """
        # Previsioni
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        # Calcolo metriche
        r2_train = ModelEvaluator.r2_score(y_train, y_train_pred)
        rmse_train = ModelEvaluator.rmse(y_train, y_train_pred)
        r2_test = ModelEvaluator.r2_score(y_test, y_test_pred)
        rmse_test = ModelEvaluator.rmse(y_test, y_test_pred)
        
        # Creazione grafico
        plt.figure(figsize=(12, 10))
        
        # Grafico dei dati e della retta di regressione
        plt.subplot(2, 1, 1)
        plt.scatter(x_train, y_train, color='blue', alpha=0.5, label='Training data')
        plt.scatter(x_test, y_test, color='green', alpha=0.5, label='Test data')
        
        # Plot della linea di regressione
        x_all = np.concatenate([x_train, x_test])
        x_line = np.array([min(x_all), max(x_all)])
        y_line = model.predict(x_line)
        plt.plot(x_line, y_line, color='red', linewidth=2, 
                 label=f'y = {model.beta_1:.2f}x + {model.beta_0:.2f}')
        
        plt.title(f'{title}\nTrain R²: {r2_train:.4f}, RMSE: {rmse_train:.4f} | Test R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}')
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
        print(f'Intercetta (β₀): {model.beta_0:.4f}')
        print(f'Pendenza (β₁): {model.beta_1:.4f}')
        print('\nMetriche sul training set:')
        print(f'R²: {r2_train:.4f}')
        print(f'RMSE: {rmse_train:.4f}')
        print('\nMetriche sul test set:')
        print(f'R²: {r2_test:.4f}')
        print(f'RMSE: {rmse_test:.4f}')


# Esempio di utilizzo
if __name__ == "__main__":
    # Generiamo dati di esempio
    x, y = Dataset.generate_sample_data(size=100, slope=2, intercept=1, noise=2, random_state=42)
    
    # Creiamo e dividiamo il dataset
    dataset = Dataset(x, y)
    (x_train, y_train), (x_test, y_test) = dataset.split(train_size=0.8, random_state=42)
    
    # Addestramento con formula chiusa
    model_closed = LinearRegressionModel()
    model_closed.fit(x_train, y_train, method=FittingMethod.CLOSED_FORM)
    
    # Visualizziamo i risultati
    Visualizer.plot_regression(x_train, y_train, x_test, y_test, model_closed, 
                              title="Regressione Lineare (Formula Chiusa)")
    
    # Addestramento con gradient descent
    model_gd = LinearRegressionModel(learning_rate=0.05, num_iterations=1000)
    model_gd.fit(x_train, y_train, method=FittingMethod.GRADIENT_DESCENT)
    
    # Visualizziamo l'andamento dell'errore
    model_gd.plot_training_history()
    
    # Visualizziamo i risultati
    Visualizer.plot_regression(x_train, y_train, x_test, y_test, model_gd, 
                              title="Regressione Lineare (Gradient Descent)")