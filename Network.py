from Layer import Layer
import numpy as np


# main Neural Network class

class Network:

    # creates a Neural Network with the specified layers 
    # so far, we only support Dense Neural Networks (due to Dense Layers)
    def __init__(self, layers_array):
        self.layers = layers_array


    # perform a forward prop on all the inputted data 
    # expects the input data in a column vector (single example) or a matrix with columns as examples
    def predict(self, input_matrix):
        input_ = input_matrix
        activation_matrix = [input_]
        for i in range(len(self.layers)):
            input_ = self.layers[i].forward_prop(input_)
            activation_matrix.insert(i+1, input_)
        return input_
    
    ## back propagation and training 

    # expect X_train to have 1 training example per column 
    # expect Y_train to have all outputs in a row vector 
    def train(self, X_train, Y_train, iterations, learning_rate):
        # training the model using the back prop algo

        error_function = lambda output, expected: output - expected

        ## we will use stochastic gradient descent in the first implementation
        for j in range(iterations):
            ## perform forward pass and store all require data 
            output = self.predict(X_train)
            
            
            ## perform loss calculations 
            # we will currently use the mean squared error built in
            if j % (iterations / 10) == 0:
                loss = np.sum((Y_train - output)**2 / 2) / output.shape[0]
                print(loss)

            
            ## perform the output layer back prop
            output_layer = self.layers[-1]

            # print("loss function for output: ", error_function(output, Y_train))
            # print(output_layer.logits)
            # print("activation_derivative for output: ", output_layer.activation.derivative(output_layer.logits))

            error_output_layer = np.multiply(error_function(output, Y_train), 
                                            output_layer.activation.derivative(output_layer.logits))
            partial_output_layer = np.dot(error_output_layer, self.layers[-2].activation_values.T)

            # print("Output Error: ", error_output_layer)
            # print("Output Partial: ", partial_output_layer)

            # perform hidden layer back prop
            error_L_plus_1 = error_output_layer
            partial_L_plus_1 = partial_output_layer

            for i in reversed(range(len(self.layers)-1)):
                current_layer_error = np.multiply(np.dot(self.layers[i+1].weights_matrix.T, error_L_plus_1),
                                            self.layers[i].activation.derivative(self.layers[i].logits))
                current_layer_partial = np.dot(current_layer_error, self.layers[i-1].activation_values.T)
                
                # update the layer in front 
                self.layers[i+1].weights_matrix = self.layers[i+1].weights_matrix - learning_rate * partial_L_plus_1 / X_train.shape[1]
                self.layers[i+1].bias_vector = self.layers[i+1].bias_vector - learning_rate * np.sum(error_L_plus_1, axis = 1, keepdims=True)  / X_train.shape[1]
                # print("Error: ", current_layer_error)
                # print("Partial: ", current_layer_partial)
                partial_L_plus_1 = current_layer_partial
                error_L_plus_1 = current_layer_error
            
            self.layers[0].weights_matrix = self.layers[0].weights_matrix - learning_rate * current_layer_partial / X_train.shape[1]
            self.layers[0].bias_vector = self.layers[0].bias_vector - learning_rate * np.sum(current_layer_error, axis = 1, keepdims=True)  / X_train.shape[1]
            # print(current_layer_error)
            # print(self.layers[0].bias_vector)
            

        #print(error_output_layer)
        #print(partial_output_layer)

        


    



    

