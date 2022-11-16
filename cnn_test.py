import gwu_nn
from gwu_nn.activation_layers import Sigmoid
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Flatten

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
 
network = GWUNetwork()

# Test Code for Flatten Layer
test_flat = np.array([[1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16]])
test_flat = Flatten.forward_propagation(network, input=test_flat)
test_flat_back = Flatten.backward_propagation(network, input=test_flat, learning_rate=0.01)
