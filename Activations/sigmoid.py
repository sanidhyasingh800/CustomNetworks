from .Activation import Activation
import numpy as np

class sigmoid(Activation):
    
    def activate(self, input_):
        return 1 / (1 + np.exp(-input_))
    
    def derivative(self, input_):
        return self.activate(input_)*(1-self.activate(input_))