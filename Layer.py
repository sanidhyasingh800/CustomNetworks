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
        self.inputs_per_unit = inputs_per_unit
        self.weights_matrix = np.random.uniform(-0.1, 0.1, (units, inputs_per_unit))
        self.bias_vector = (np.zeros(units)).reshape([-1,1])
        self.logits = np.array([])
        self.activation_values = np.array([])


    # perform forward prop on the given input_matrix
    # input_matrix is expected to be oriented as: 1 training example/ testing example per column
    def forward_prop(self, input_matrix):
        input_ = np.dot(self.weights_matrix, input_matrix)
        input_ = input_ + self.bias_vector
        self.logits = input_
        self.activation_values = self.activation.activate(input_)
        return self.activation_values
    
    def set_weights(self, weights_matrix):
        assert self.units == weights_matrix.shape[0], "Incorrect row size of Weight Matrix"
        assert self.inputs_per_unit == weights_matrix.shape[1], "Incorrect column size of Weight Matrix"
        self.weights_matrix = weights_matrix
    
    def set_bias(self, bias_vector):
        assert self.units ==  bias_vector.shape[0], "Incorrect size of bias vector"
        self.bias_vector = bias_vector

    # return the weight matrix and bias vector as a couple
    def get_params(self):
        return self.weights_matrix, self.bias_vector

    