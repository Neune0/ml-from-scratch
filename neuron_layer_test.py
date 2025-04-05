from neuron_layer import NeuroLayer
from activation import relu
def __main__():
    l1 = NeuroLayer(5,relu,0.1,3)
    l_input = [0.5,0.2,0.1]
    out = l1.process(l_input)
    print(out)
    
if __name__ == "__main__":
    __main__()