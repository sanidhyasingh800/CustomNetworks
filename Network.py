from Layer import Layer
import numpy as np


# main Neural Network class

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
    def train(self, X_train, Y_train, epochs, learning_rate, batch_size):
        # training the model using the back prop algo

        error_function = lambda output, expected: output - expected

        ## we will use stochastic gradient descent in the first implementation
        for j in range(epochs):
            for batch in range(0, X_train.shape[0], batch_size):
                current_batch_X= X_train[batch: batch + batch_size]
                current_batch_Y = Y_train[batch: batch + batch_size].T
                # print(current_batch_X)
                # print(current_batch_Y)
                ## perform forward pass and store all require data 
                output = self.predict(current_batch_X).T
                # print(output)
                
                
                ## perform loss calculations 
                # we will currently use the mean squared error built in
                if j % (epochs / 10) == 0:
                    loss = np.sum((current_batch_Y - output)**2 / 2) / output.shape[0]
                    print("Loss: ",  loss)

                for i in reversed(range(len(self.layers))):
                    # Error Calculation for Output Layer Using L2 loss
                    if i == len(self.layers)-1:
                        current_layer_error = np.multiply(error_function(output, current_batch_Y), 
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


        


    



    
