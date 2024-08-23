import numpy as np
from .Loss import Loss
class BinaryCrossEntropy(Loss):

    # returns the mean Binary Cross loss over the batch
    def calculate_loss(self, expected, output):
        epsilon = 0.0000000001
        return -np.mean(expected*np.log(output + epsilon) + (1-expected) * np.log(1-output + epsilon), axis = 0)
    
    # returns the derivative of BinaryCrossEntropy loss
    # used in backpropagation of the output layer
    # derivative taken w.r.t to the output term
    # returns the derivative for each example along rows and for each weight along columns
    def loss_derivative(self, expected, output):
        epsilon =  0.0000000001
        return  (output-expected)/ (output * (1-output) + epsilon)
