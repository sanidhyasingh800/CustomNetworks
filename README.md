# Custom Neural Network Library [Inspired by Tensorflow]

## Table of Contents
1. [Introduction](#Introduction)
2. [Documentation](#Documentation)
4. [Example](#example)



## Introduction
A custom neural network library for dense NNs inspired by the modularity in tensorflow. Design is super beginner friendly and allows for custom architectures, activations, and training mechanisms.

## Documentation

### Network Class

#### 1. `Network(layer_list)`
**Description:**  
Constructs an empty network based on the specified layers.

**Parameters:**
- `layer_list` (list): A list of layers that define the structure of the network.

#### 2. `NN.train(self, X_train, Y_train, epochs=5, learning_rate=0.1, batch_size=32, loss=L2(), display_losses=False)`
**Description:**  
Trains the network using backpropagation and gradient descent.

**Parameters:**
- `X_train` (np.array): Matrix of training data where each row is a sample and each column is a feature.
- `Y_train` (np.array):  Matrix of labels corresponding to the training data where each column is a different label.
- `epochs` (int, default=5): Number of training iterations.
- `learning_rate` (float, default=0.1): Learning rate for gradient descent.
- `batch_size` (int, default=32): Size of each mini-batch.
- `loss` (object, default=L2()): Loss function used for training.
- `display_losses` (bool, default=False): If `True`, displays loss during training.

#### 3. `NN.predict(X_test)`
**Description:**  
Returns the predicted labels corresponding to X_test

**Parameters:**
- `X_test` (np.array) :Matrix of testing data where each row is a sample and each column is a feature. **Must match the same number of features as training data.**

### Layer Class

#### 1. `Layer(units, activation, inputs_per_unit)`
**Description:**  
Initializes a new dense layer with the specified number of units, activation function, and inputs per unit.

**Parameters:**
- `units` (int): Number of neurons in this layer.
- `activation` (object): Activation function class used in this layer (`relu()`, `sigmoid()`, `linear()`, or `tanh()`).
- `inputs_per_unit` (int): Number of inputs each unit receives (equal to the number of units in the previous layer).

#### 2. `L.forward_prop(input_matrix)`
**Description:**  
Performs forward propagation on the given input matrix and returns the result. The method calculates the weighted input (logits) by multiplying the input matrix by the weights, adding the bias, and then applying the activation function.

**Parameters:**
- `input_matrix` (ndarray): The input data for the layer. **It is expected to be oriented as one training/testing example per column. This is different from the Network Class!**

#### 3.`L.set_weights(weights_matrix)`
**Description:**  
Sets the weight matrix for the layer. Ensures that the dimensions of the new weight matrix match the layer's configuration, throws an error otherwise.

**Parameters:**
- `weights_matrix` (ndarray): The new weight matrix with shape `(units, inputs_per_unit)` or each neuron's weight in a single row.

#### 4.`L.set_buas(bias_vector)`
**Description:**  
Sets the bias_vector for the layer. Ensures that the dimensions of the new bias vector is valid, throws an error otherwise.

**Parameters:**
- `bias_vector` (ndarray): The new bias vector with shape `(units, 1)` or each neuron's bias in a single row.

#### 5. `L.get_params()`
**Description:**  
Returns the current weight matrix and bias vector as a tuple. This method is useful for inspecting the layer's parameters or for saving them.

### Activations:
- `linear()`
- `sigmoid()`
- `relu()`
- `tanh()`

### Loss:
- `L1()`
- `BinaryCrossEntropy()`

## Example
```python
NN = Network([Layer(units = 500, activation=relu(), inputs_per_unit=784),
              Layer(units = 250, activation=relu(), inputs_per_unit=500),
                Layer(units = 100, activation = relu(), inputs_per_unit = 250), 
                Layer(units = 10, activation=sigmoid(), inputs_per_unit=100)])
NN.train(X_train_flattened, y_train, 
         epochs = 5, learning_rate = 0.1, batch_size = 32, 
         loss = BinaryCrossEntropy(), display_losses = True)

output = NN.predict(X_test_flattened)

```