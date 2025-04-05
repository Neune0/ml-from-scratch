from neuron import Neuron

class NeuroLayer:
    def __init__(self,numNeurons,activationFunc, bias,dimPrevLayer):
        self.neurons = [Neuron(dimPrevLayer,activationFunc,bias) for _ in range(numNeurons)]
    
    def process(self,inputs):
        return [neuron.process(inputs) for neuron in self.neurons]          