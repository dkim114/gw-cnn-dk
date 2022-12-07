import gwu_nn
from gwu_nn.activation_layers import Sigmoid
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Flatten, Conv2D, MaxPooling2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist 


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train_subset = []
y_train_subset = []

x_test_subset = []
y_test_subset = []

for i in range (0, 500):
    if (y_train[i] == 0 or y_train[i] == 1):
        x_train_subset.append(x_train[i])
        y_train_subset.append(y_train[i])

for i in range (0, 500):
    if (y_test[i] == 0 or y_test[i] == 1):
        x_test_subset.append(x_test[i])
        y_test_subset.append(y_test[i])

x_train_subset = np.array(x_train_subset[:5])
x_test_subset = np.array(x_test_subset[:5])
y_train_subset = np.array(y_train_subset[:5])
y_test_subset = np.array(y_test_subset[:5])

x_train_subset = x_train_subset.reshape(x_train_subset.shape[0], 28, 28, 1).astype('float32')
x_test_subset = x_test_subset.reshape(x_test_subset.shape[0], 28, 28, 1).astype('float32')


print("x_train_subset Shape: " + str(x_train_subset.shape))
print("y_train_subset Shape: " + str(y_train_subset.shape))
print("x_test_subset Shape: " + str(x_test_subset.shape))
print("y_test_subset Shape: " + str(y_test_subset.shape))

network = GWUNetwork()
network.add(Conv2D(input_size=28, kernel_size=3))
network.add(MaxPooling2D(pool_size=2, strides=2, input_size=26))
network.add(Flatten(input_size=(13,13)))
network.add(Dense(1, input_size=13**2, add_bias=False, activation='sigmoid'))
network.compile(loss='mse', lr=0.01)
print(network)
network.fit(x_train_subset, y_train_subset, epochs=1)
results = network.predict(x_test_subset)
print(results)

d_round = lambda x: 1 if x >= 0.5 else 0
predictions = [d_round(x[0]) for x in results]
actual = [y for y in y_test_subset.reshape(-1)]

print(predictions)
print(actual)

# Random test code
#print(np.array([x_train.flatten()]).shape)

#network.fit(x_train, y_train, epochs=100, batch_size=20)

# First Layer: Conv2D(num_filters, kernel_size, activation, input_shape)
# Second Layer: MaxPooling2D(pool_size)
# Third Conv2D Layer: Conv2D(num_filters, kernel_size, activation)
# Fourth Layer: MaxPooling2D(pool_size)
# Fifth Layer: Flatten()

# Test Code for Flatten Layer
#test_flat = np.array([[1,2,3,4],
#                    [5,6,7,8],
#                    [9,10,11,12],
#                    [13,14,15,16]])
#test_flat = Flatten.forward_propagation(network, input=test_flat)
#test_flat_back = Flatten.backward_propagation(network, input=test_flat, learning_rate=0.01)


