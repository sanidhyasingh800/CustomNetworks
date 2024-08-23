from .Activation import Activation
import numpy as np

class tanh(Activation):
    
    def activate(self, input_):
        return (np.exp(input_) - np.exp(-input_)) / (np.exp(input_) + np.exp(-input_))
    
    def derivative(self, input_):
        return 1 - self.activate(input_)**2