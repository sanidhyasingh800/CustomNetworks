from Layer import Layer
from Activations.sigmoid import sigmoid
from Activations.relu import relu
from Activations.linear import linear
from Network import Network
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np
from colorama import Fore


(X_train, y_train), (X_test, y_test) = mnist.load_data()
# # Filter the training set
# train_filter = np.where((y_train == 0) | (y_train == 1))
# X_train, y_train = X_train[train_filter], y_train[train_filter]

# # Filter the test set
# test_filter = np.where((y_test == 0) | (y_test == 1))
# X_test, y_test = X_test[test_filter], y_test[test_filter]

# Flatten the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1).T
X_test_flattened = X_test.reshape(X_test.shape[0], -1).T
y_train = to_categorical(y_train, 10).T

# Normalize the pixel values to [0, 1]
# X_train_flattened = X_train_flattened / 255.0
# X_test_flattened = X_test_flattened / 255.0
print(X_train_flattened.shape)
print(y_train)


NN = Network([Layer(units = 25, activation=relu(), inputs_per_unit=784),
              Layer(units = 20, activation=relu(), inputs_per_unit=25),
                Layer(units = 15, activation = relu(), inputs_per_unit = 20), 
                Layer(units = 10, activation=linear(), inputs_per_unit=15)])



NN.train(X_train_flattened, y_train, iterations = 300, learning_rate = 0.1)
NN.train(X_train_flattened, y_train, iterations = 1000, learning_rate = 0.01)
NN.train(X_train_flattened, y_train, iterations = 100, learning_rate = 0.1)
output = NN.predict(X_test_flattened)[0]
count = 0
for i in range(200):
    # if y_test[i] == 1 and output[i] > 0.50:
    #     count = count + 1
    # if y_test[i] == 0 and output[i] < 0.50:
    #     count = count + 1
    print(output[i], 'expected: ', y_test[i])

print(count)

#for i in range(len(NN.layers)):
 #   print(NN.layers[i].weights_matrix)9