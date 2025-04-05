import numpy as np

def plain(x):
    return x

# non differenziabile
def step(x,soglia):
    return 1 if x > soglia else 0

# differenziabile
def relu(x,soglia):
    return x if x > soglia else 0

# differenziabile
def relu(x):
    return x if x > 0 else 0

# differenziabile
def tanh(x):
    return np.tanh(x)

# differenziabile
def sigmoid(x):
    return 1/(1+np.exp(-x))
