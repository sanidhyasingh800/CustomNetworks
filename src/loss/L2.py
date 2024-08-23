import numpy as np
from .Loss import Loss
class L2(Loss):

    # returns the mean L2 loss over the batch
    def calculate_loss(self, expected, output):
        return np.mean((expected - output)**2, axis = 0)
    
    # returns the derivative of L2 loss
    # used in backpropagation of the output layer
    # derivative taken w.r.t to the output term
    # returns the derivative for each example along rows and for each weight along columns
    def loss_derivative(self, expected, output):
        return 2 *(output - expected)
