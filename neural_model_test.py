from neural_model import NeuralModel
import numpy as np
from activation import relu,sigmoid
def __main__():
    num_layers = 10
    numNeuronPerLayer = [num_layers-i for i in range(num_layers)]
    activations = [sigmoid for _ in range(num_layers)]
    init_bias = [0.2 for _ in range(num_layers)]
    neuralModel = NeuralModel(num_layers,numNeuronPerLayer,activations,init_bias,10)
    initialInput = [np.random.rand() for _ in range(num_layers)]
    
    
    print(neuralModel.process(initialInput))
    return 0

if __name__ == "__main__":
    __main__()