class Neuron:
    
    # costruttore
    def __init__(self,dim_input,attivation_func,bias):
        self.dim_input= dim_input
        self.attivation_func = attivation_func
        self.bias = bias
        self.weights = [0.5 for _ in range(dim_input)]
    
    # sommatore
    def _sommatore(self, input):
        return sum([self.weights[i] * input[i] for i in range(self.dim_input)]) + self.bias
    
    def process(self,input):
        return self.attivation_func(self._sommatore(input))
    
    
    