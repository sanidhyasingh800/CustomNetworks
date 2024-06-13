from Layer import Layer
from Activations.sigmoid import sigmoid
from Activations.relu import relu
from Network import Network


import numpy as np


# NN = Network([Layer(units = 2, activation=sigmoid(), inputs_per_unit=2), 
#               Layer(units = 1, activation=sigmoid(), inputs_per_unit=2)])



NN = Network([Layer(units = 4, activation=relu(), inputs_per_unit=2),
                Layer(units = 2, activation = relu(), inputs_per_unit = 4), 
                Layer(units = 1, activation=sigmoid(), inputs_per_unit=2)])

# NN = Network([Layer(units = 4, activation=relu(), inputs_per_unit=2), 
#                 Layer(units = 1, activation=sigmoid(), inputs_per_unit=4)])

x_input = np.array([[0.35, 0.9]]).T
y = np.array([[1]])
print(NN.predict(x_input))

NN.train(x_input, y, iterations = 10000, learning_rate = 0.01)
print(NN.predict(x_input))

#for i in range(len(NN.layers)):
 #   print(NN.layers[i].weights_matrix)
