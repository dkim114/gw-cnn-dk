import gwu_nn
from gwu_nn.activation_layers import Sigmoid
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_subset = []
y_train_subset = []

x_test_subset = []
y_test_subset = []

print(x_train.shape)

for i in range (0, 2000):
    if (y_train[i] == 0 or y_train[i] == 1):
        x_train_subset.append(x_train[i])
        y_train_subset.append(y_train[i])

for i in range (0, 2000):
    if (y_test[i] == 0 or y_test[i] == 1):
        x_test_subset.append(x_test[i])
        y_test_subset.append(y_test[i])

x_train_subset = np.array(x_train_subset[:500])
x_test_subset = np.array(x_test_subset[:10])
y_train_subset = np.array(y_train_subset[:500])
y_train_subset = np.array(tf.keras.utils.to_categorical(y_train_subset, num_classes=2))
y_test_subset = np.array(y_test_subset[:10])

x_train_subset = x_train_subset.reshape(x_train_subset.shape[0], 28, 28).astype('float32')
x_test_subset = x_test_subset.reshape(x_test_subset.shape[0], 28, 28).astype('float32')
x_train_subset /= 255.0
x_test_subset /= 255.0

print("x_train_subset Shape: " + str(x_train_subset.shape))
print("y_train_subset Shape: " + str(y_train_subset.shape))
print("x_test_subset Shape: " + str(x_test_subset.shape))
print("y_test_subset Shape: " + str(y_test_subset.shape))

network = GWUNetwork()
network.add(Conv2D(input_size=28, kernel_size=3))
network.add(MaxPooling2D(pool_size=3, strides=2, input_size=23))
network.add(Flatten(input_size=(11,11)))
network.add(Dense(100, add_bias=False, activation='relu'))
network.add(Dense(2, add_bias=False, activation='sigmoid'))
network.compile(loss='log_loss', lr=0.001)
print(network)
network.fit(x_train_subset, y_train_subset, epochs=1)
results = network.predict(x_test_subset)
print(str(results) + "\n")

#d_round = lambda x: 1 if x > 0.5 else 0
#predictions = [d_round(x[0]) for x in results]
actual = [y for y in y_test_subset.reshape(-1)]
print('Actual: ' + str(actual))

temp = np.exp(results)
results = temp / np.sum(temp)

predic = []
for x in results:
    if x[0][0] > x[0][1]:
        predic.append(0)
    else:
        predic.append(1)

print('Predic: ' + str(predic))


