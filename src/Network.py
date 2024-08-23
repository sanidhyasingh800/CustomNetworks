from .Layer import Layer
from .loss import *
import numpy as np


# main Neural Network class
"""
Note: User passes the data as:
    input : 1 example per row
    output: 1 output per row (if a NN produces multiple outputs, all outputs corresponding to 1 example per row)

For passes between layers, this is transposed
    the output of each layer is 1 example per column 

"""



class Network:

    # creates a Neural Network with the specified layers 
    # so far, we only support Dense Neural Networks (due to Dense Layers)
    def __init__(self, layers_array):
        self.layers = layers_array


    # perform a forward prop on all the inputted data 
    # expects the input data in a row vector (single example) or a matrix with rows as examples
    def predict(self, input_matrix):
        input_ = input_matrix.T
        activation_matrix = [input_]
        for i in range(len(self.layers)):
            input_ = self.layers[i].forward_prop(input_)
            activation_matrix.insert(i+1, input_)
        return input_.T # returns a single output per row
    
    ## back propagation and training 

    # expect X_train to have 1 training example per row 
    # expect Y_train to have all outputs in a row vector 
    def train(self, X_train, Y_train, epochs = 5, learning_rate = 0.1, batch_size = 32, loss = L2(), display_losses = False):
        # training the model using back prop

        # error_function = lambda output, expected: output - expected


        for j in range(epochs):
            for batch in range(0, X_train.shape[0], batch_size):
                current_batch_X= X_train[batch: batch + batch_size]
                current_batch_Y = Y_train[batch: batch + batch_size].T # transpose so that the format of the output matches the format of data passed through the network
                # print(current_batch_X)
                # print(current_batch_Y)
                ## perform forward pass and store all require data 
                output = self.predict(current_batch_X).T # since self.predict returns outputs for a single example per row, but our network uses single example per column
                # print("output: ", output)
                
                
                ## perform loss calculations 
                # we will currently use the mean squared error built in
                if display_losses and j % (epochs / 10) == 0:
                    current_loss = np.sum(loss.calculate_loss(current_batch_Y.T, output.T))
                    # np.sum((current_batch_Y - output)**2 / 2) / output.shape[1]
                    print("Loss: ",  current_loss)

                for i in reversed(range(len(self.layers))):
                    # Error Calculation for Output Layer Using user provided loss function
                    if i == len(self.layers)-1:
                        current_layer_error = np.multiply(loss.loss_derivative(current_batch_Y.T, output.T).T, 
                                                self.layers[i].activation.derivative(self.layers[i].logits))
                    # hidden layer error calculation
                    else: 
                        current_layer_error = np.multiply(np.dot(self.layers[i+1].weights_matrix.T, error_L_plus_1),
                                                self.layers[i].activation.derivative(self.layers[i].logits))

                    # initial layer partial is calculated using the inputs
                    if i == 0:
                        current_layer_partial = np.dot(current_layer_error, current_batch_X)
                    # hidden layer partials are calculated using the previous layer's activations 
                    else:
                        current_layer_partial = np.dot(current_layer_error, self.layers[i-1].activation_values.T)
                    

                    # weight updates 
                    if i != len(self.layers)-1:
                        # update the layer in front 
                        self.layers[i+1].weights_matrix = self.layers[i+1].weights_matrix - learning_rate * partial_L_plus_1 / X_train.shape[1]
                        self.layers[i+1].bias_vector = self.layers[i+1].bias_vector - learning_rate * np.sum(error_L_plus_1, axis = 1, keepdims=True)  / X_train.shape[1]
                    if i == 0:
                        self.layers[0].weights_matrix = self.layers[0].weights_matrix - learning_rate * current_layer_partial / X_train.shape[1]
                        self.layers[0].bias_vector = self.layers[0].bias_vector - learning_rate * np.sum(current_layer_error, axis = 1, keepdims=True)  / X_train.shape[1]
                    
                    # back propagate the partials and error
                    partial_L_plus_1 = current_layer_partial
                    error_L_plus_1 = current_layer_error


        


    



    

