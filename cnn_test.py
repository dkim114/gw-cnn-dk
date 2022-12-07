import gwu_nn
from gwu_nn.activation_layers import Sigmoid
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Flatten, Conv2D, MaxPooling2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = x_train[:1]
y_train = y_train[:1]
print(x_train.shape)
print(y_train.shape)
print(y_train)

network = GWUNetwork()
network.add(Conv2D(input_size=28, kernel_size=5))
# Max Pooling
# Conv2D
# Max Pooling
network.add(MaxPooling2D(pool_size=2, strides=2, input_size=26))
#network.add(Flatten(input_size=(24,24)))
#network.add(Dense(1, activation='relu', input_size=1152))
#network.add(Sigmoid())
network.compile(loss='log_loss', lr=0.01)
print(network)
network.fit(x_train, y_train, epochs=1)

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


