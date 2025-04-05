from neuron_layer import NeuroLayer
class NeuralModel:
    def __init__(self,numLayer:int,numNeuronsPerLayer:list[int], activationPerLayer:list[callable],initBiasPerLayer:list[int],dimInput):
        
        self.layers = [NeuroLayer(numNeuronsPerLayer[i],activationPerLayer[i],initBiasPerLayer[i],numNeuronsPerLayer[i-1] if i-1> 0 else dimInput) for i in range(numLayer)]
    
    def process(self,inputs):
        # inputs passa al layer 1 che processa e passa la layer 2 ecc
        for layer in self.layers:
            inputs = layer.process(inputs)
        return inputs