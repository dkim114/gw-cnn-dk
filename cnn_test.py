import gwu_nn
from gwu_nn.activation_layers import Sigmoid
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Flatten, Conv2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist
 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

print(x_train.shape)
print(y_train.shape)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = x_train[:1]
y_train = y_train[:1]
print(x_train.shape)
print(y_train.shape)


network = GWUNetwork()
#network.add(Dense(output_size=16, input_size=5))
network.add(Conv2D(input_size=28, kernel_size=5))
network.add(Flatten(input_size=(24,24)))
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


