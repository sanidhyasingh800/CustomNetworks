from CustomNetworks import *
from tensorflow.keras.datasets import mnist
import numpy as np


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# One hot encode
y_train = np.eye(np.max(y_train) + 1)[y_train]
y_test = np.eye(np.max(y_test) + 1)[y_test]

#Normalize the pixel values to [0, 1]
X_train_flattened = (X_train_flattened - np.mean(X_train_flattened, axis = 0)) / (np.std(X_train_flattened, axis = 0) + 0.00000001)
X_test_flattened = (X_test_flattened - np.mean(X_test_flattened, axis = 0)) / (np.std(X_test_flattened, axis = 0) + 0.00000001)


NN = Network([Layer(units = 500, activation=relu(), inputs_per_unit=784),
              Layer(units = 250, activation=relu(), inputs_per_unit=500),
                Layer(units = 100, activation = relu(), inputs_per_unit = 250), 
                Layer(units = 10, activation=sigmoid(), inputs_per_unit=100)])
NN.train(X_train_flattened, y_train, 
         epochs = 5, learning_rate = 0.1, batch_size = 32, 
         loss = BinaryCrossEntropy(), display_losses = True)

output = NN.predict(X_test_flattened)
count = 0
for i in range(y_test.shape[0]):
    if np.argmax(output[i]) == np.argmax(y_test[i]):
        count = count + 1
    print(np.argmax(output[i]), 'expected: ', np.argmax(y_test[i]))

print(count, " / ", y_test.shape[0])


