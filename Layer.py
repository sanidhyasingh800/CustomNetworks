import numpy as np
## Dense Layer Class
class Layer:

    # create a new dense layer with the given number of units and each with the activation function
    def __init__(self, units, activation, inputs_per_unit):
        # units is the number of neurons within the layer 
        # activation is the function class we use to perform forward prop
        # inputs_per_unit is the number of outputs (or units) in the previous layer
        #   when linking together: assert(layers prev_layer.units == next_layer.inputs_per_unit) for dense NNs
        self.units = units
        self.activation = activation
        self.weights_matrix = np.zeros((units,inputs_per_unit))
        self.bias_vector = (np.zeros(units)).reshape([-1,1])


    # perform forward prop on the given input_matrix
    # input_matrix is expected to be oriented as: 1 training example/ testing example per column
    def forward_prop(self, input_matrix):
        input_ = np.dot(self.weights_matrix, input_matrix) + self.bias_vector.reshape([-1,1])
        return self.activation(input_)
    
    def set_weights(self, weights_matrix):
        # assert(self.units = weight_matrix.shape[0])
        # assert(self.inputs_per_unit = weight_matrix.shape[1])
        self.weights_matrix = weights_matrix
    
    def set_bias(self, bias_vector):
        # assert(self.units = bias_vector.shape[0])
        self.bias_vector = bias_vector

    # return the weight matrix and bias vector as a couple
    def get_params(self):
        return self.weights_matrix, self.bias_vector

    