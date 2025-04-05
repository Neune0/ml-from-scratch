from neuron import Neuron
import argparse

def __main__():
    parser = argparse.ArgumentParser(description="Test di un neurone artificiale")
    parser.add_argument("--bias",type=float,default=0.1,help="Valore del bias (default: 0.1)")
    parser.add_argument("--inputs", type=float, nargs=3, default=[0.2, 0.4, 0.7], 
                        help="3 valori di input separati da spazio (default: 0.2 0.4 0.7)")
    
    args = parser.parse_args()
    
    neuron = Neuron(3, lambda a: a, bias=args.bias)
    print(f"Bias: {args.bias}")
    print(f"Inputs: {args.inputs}")
    print(f"Output: {neuron.process(args.inputs)}")

if __name__ == "__main__":
    __main__()