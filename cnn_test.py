import gwu_nn
from gwu_nn.activation_layers import Sigmoid
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Flatten

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist
 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

network = GWUNetwork()
network.add(Flatten(input_size=(28,28)))
network.add(Sigmoid())
network.compile(loss='log_loss', lr=0.01)
print(network)

# Random test code
print(np.array([x_train.flatten()]).shape)

#network.fit(x_train, y_train, epochs=100, batch_size=20)

# Test Code for Flatten Layer
#test_flat = np.array([[1,2,3,4],
#                    [5,6,7,8],
#                    [9,10,11,12],
#                    [13,14,15,16]])
#test_flat = Flatten.forward_propagation(network, input=test_flat)
#test_flat_back = Flatten.backward_propagation(network, input=test_flat, learning_rate=0.01)
