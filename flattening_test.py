import gwu_nn
from gwu_nn.activation_layers import Sigmoid
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Flatten, Conv2D, MaxPooling2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_arr = np.array([[[1,6,3,4],
                      [5,2,7,8],
                      [9,10,11,12],
                      [13,14,16,15]],
                      [[1,6,3,4],
                      [5,2,7,8],
                      [9,10,11,12],
                      [13,14,16,15]]])
test_arr2 = np.array([[1,2,3],
                      [5,6,7],
                      [9,10,11]])
                      

test_filter = np.array([[1,0],[1,0]])

test_layer = Flatten(input_size=(4,4))
test_forward = test_layer.forward_propagation(test_arr)
test_backward = test_layer.backward_propagation(test_arr, 0.001)

print(test_arr)
print(test_forward[0])
print(test_backward)



