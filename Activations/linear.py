from .Activation import Activation
import numpy as np

class linear(Activation):
    
    def activate(self, input_):
        return input_
    
    def derivative(self, input_):
        return 1