from Layer import Layer
from Activations.sigmoid import sigmoid
from Activations.linear import linear

from Activations.relu import relu
from Network import Network
import tensorflow
from tensorflow.keras.datasets import mnist

import numpy as np
import seaborn as sns
from colorama import Fore


(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Filter the training set
# train_filter = np.where((y_train == 0) | (y_train == 1) | (y_train == 2))
# X_train, y_train = X_train[train_filter], y_train[train_filter]

# # Filter the test set
# test_filter = np.where((y_test == 0) | (y_test == 1)  | (y_test == 2))
# X_test, y_test = X_test[test_filter], y_test[test_filter]

# Flatten the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# One hot encode
y_train = np.eye(np.max(y_train) + 1)[y_train]
y_test = np.eye(np.max(y_test) + 1)[y_test]


#Normalize the pixel values to [0, 1]
X_train_flattened = (X_train_flattened - np.mean(X_train_flattened, axis = 0)) / (np.std(X_train_flattened, axis = 0) + 0.00000001)
X_test_flattened = (X_test_flattened - np.mean(X_test_flattened, axis = 0)) / (np.std(X_test_flattened, axis = 0) + 0.00000001)
print(X_train_flattened.shape)
print(y_train)


NN = Network([Layer(units = 500, activation=relu(), inputs_per_unit=784),
              Layer(units = 250, activation=relu(), inputs_per_unit=500),
                Layer(units = 100, activation = relu(), inputs_per_unit = 250), 
                Layer(units = 10, activation=sigmoid(), inputs_per_unit=100)])


# NN = Network([Layer(units = 4, activation=relu(), inputs_per_unit=784),
#               Layer(units = 3, activation=sigmoid(), inputs_per_unit=4)])

NN.train(X_train_flattened, y_train, epochs = 20, learning_rate = 0.1, batch_size = 32)
# NN.train(X_train_flattened, y_train, iterations = 1000, learning_rate = 0.01)
# NN.train(X_train_flattened, y_train, iterations = 100, learning_rate = 0.1)
output = NN.predict(X_test_flattened)
count = 0
for i in range(2000):
    # if y_test[i] == 1 and output[i] > 0.50:
    #     count = count + 1
    if np.argmax(output[i]) == np.argmax(y_test[i]):
        count = count + 1
    print(np.argmax(output[i]), 'expected: ', np.argmax(y_test[i]))

print(count)
print(y_test)
print(output)


# for i in range(len(NN.layers)):
#    print(NN.layers[i].weights_matrix)


