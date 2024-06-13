from .Activation import Activation
import numpy as np

class relu(Activation):
    
    def activate(self, input_):
        return np.maximum(0, input_)
    
    def derivative(self, input_):
        return np.where(input_>0, 1, 0)