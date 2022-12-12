import numpy as np
from abc import ABC, abstractmethod
from gwu_nn.activation_layers import Sigmoid, RELU, Softmax

activation_functions = {'relu': RELU, 'sigmoid': Sigmoid, 'softmax': Softmax}

import matplotlib.pyplot as plt
from numpy import unravel_index
from scipy.signal import convolve2d

def apply_activation_forward(forward_pass):
    """Decorator that ensures that a layer's activation function is applied after the layer during forward
    propagation.
    """
    def wrapper(*args):
        output = forward_pass(args[0], args[1])
        if args[0].activation:
            return args[0].activation.forward_propagation(output)
        else:
            return output
    return wrapper


def apply_activation_backward(backward_pass):
    """Decorator that ensures that a layer's activation function's derivative is applied before the layer during
    backwards propagation.
    """
    def wrapper(*args):
        output_error = args[1]
        learning_rate = args[2]
        if args[0].activation:
            output_error = args[0].activation.backward_propagation(output_error, learning_rate)
        return backward_pass(args[0], output_error, learning_rate)
    return wrapper


class Layer():
    """The Layer layer is an abstract object used to define the template
    for other layer types to inherit"""

    def __init__(self, activation=None):
        """Because Layer is an abstract object, we don't provide any detailing
        on the initializtion"""
        self.type = "Layer"
        if activation:
            self.activation = activation_functions[activation]()
        else:
            self.activation = None

    @apply_activation_forward
    def forward_propagation(cls, input):
        """:noindex:"""
        pass

    @apply_activation_backward
    def backward_propogation(cls, output_error, learning_rate):
        """:noindex:"""
        pass


class Dense(Layer):
    """The Dense layer class creates a layer that is fully connected with the previous
    layer. This means that the number of weights will be MxN where M is number of
    nodes in the previous layer and N = number of nodes in the current layer.
    """

    def __init__(self, output_size, add_bias=False, activation=None, input_size=None):
        super().__init__(activation)
        self.type = None
        self.name = "Dense"
        self.input_size = input_size
        self.output_size = output_size
        self.add_bias = add_bias


    def init_weights(self, input_size):
        """Initialize the weights for the layer based on input and output size

        Args:
            input_size (numpy array): dimensions for the input array
        """
        if self.input_size is None:
            self.input_size = input_size

        self.weights = np.random.randn(input_size, self.output_size) / np.sqrt(input_size + self.output_size)

        # TODO: Batching of inputs has broken how bias works. Need to address in next iteration
        if self.add_bias:
            self.bias = np.random.randn(1, self.output_size) / np.sqrt(input_size + self.output_size)


    @apply_activation_forward
    def forward_propagation(self, input):
        """Applies the forward propagation for a densely connected layer. This will compute the dot product between the
        input value (calculated during forward propagation) and the layer's weight tensor.

        Args:
            input (np.array): Input tensor calculated during forward propagation up to this layer.

        Returns:
            np.array(float): The dot product of the input and the layer's weight tensor."""
        self.input = input
        output = np.dot(input, self.weights)

        if self.add_bias:
            return output + self.bias
        else:
            return output
        
    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        """Applies the backward propagation for a densely connected layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         weights (dot product of the layer's input values and the output_error).

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        if self.add_bias:
            self.bias -= learning_rate * output_error
        
        return input_error

class Conv2D(Layer):

    def __init__(self, input_size=None, kernel_size=None, activation=None, depth=2):
        super().__init__(None)
        self.name = "Conv2D"
        self.input_size = input_size
        self.input_shape = (self.input_size, self.input_size, 1)
        self.kernel_size = kernel_size
        self.output_size = (((input_size - kernel_size) + 1), ((input_size - kernel_size) + 1))
        self.depth = depth

    def init_weights(self, input_size):
        # Initialize weights of kernel of size (kernel_size, kernel_size)
        self.kernel_shape = (self.depth, self.kernel_size, self.kernel_size)
        self.kernel = np.random.randn(self.depth, self.kernel_size, self.kernel_size)
    
    @apply_activation_forward
    def forward_propagation(self, input):

        #print("Conv2D Input Shape: " + str(input.shape))
        self.input = input
        currentKernelSize = self.kernel_size

        convRow = (self.input_size - currentKernelSize) + 1 # Number of rows after convolved
        convColumn = (self.input_size - currentKernelSize) + 1 # Number of columns after convolved
        self.convolve_size = convRow

        output = np.zeros((self.depth, convRow, convColumn))

        for i in range (0, self.depth):
            for x in range(0, convRow):
                for y in range (0, convColumn):
                    for z in range(0, self.kernel_size):
                        for v in range (0, self.kernel_size):
                            output[i, x, y] += input[x + z, y + v] * self.kernel[i, z, v]
        #print("Conv2D Output Shape: " + str(output.shape))
        return output

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        #print("Original Filter: \n" + str(self.filters))
        
        kernel_gradient = np.zeros(self.kernel_shape) # (2, 3, 3)
        input_gradient = np.zeros(self.input.shape)

        # Convolution
        for i in range (0, self.depth):
            # Note: Need to implement convolution between output_error and kernel
            # Note: scipy convole2d is temporary
            input_gradient += convolve2d(output_error[i], self.kernel[i])
            for x in range (0, self.kernel_size):
                for y in range (0, self.kernel_size):
                    for z in range (0, self.convolve_size):
                        for v in range (0, self.convolve_size):
                            kernel_gradient[i, x, y] += self.input[x + z, y + v] * output_error[i, z, v]

        # Update Filters with Learning Rate
        self.kernel -= np.array(kernel_gradient) * learning_rate
        return input_gradient

class MaxPooling2D(Layer):

    def __init__(self, pool_size=2, strides=1, activation=None, input_size=None):
        super().__init__(None)
        self.name = "MaxPooling2D"
        self.input_size = input_size
        self.pool_size = pool_size
        self.strides = strides
        self.output_size = (int((self.input_size - self.pool_size) / self.strides) + 1)

    def init_weights(self, input_size):
        pass

    @apply_activation_forward
    def forward_propagation(self, input):
        self.input = input
        self.input_shape = input.shape
        #print("MaxPooling2D Input Shape: " + str(input.shape))
        self.num_filters = input.shape[0]
        tempOutputSize = self.output_size
        output = np.zeros((self.num_filters, self.output_size, self.output_size))
        
        for i in range (0, self.num_filters):
            for x in range (0, tempOutputSize):
                for y in range (0, tempOutputSize):
                    tempArray = input[i, x*self.strides:(x*self.strides)+self.pool_size, y*self.strides:(y*self.strides)+self.pool_size]
                    output[i, x, y] = np.max(tempArray)

        #print("MaxPooling2D Output Shape: " + str(output.shape))
        return output

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        
        input_gradient = np.zeros(self.input_shape)
        
        for i in range (0, self.num_filters):
            y_coord = 0
            for x in range (0, self.output_size):
                x_coord = 0
                for y in range (0, self.output_size):
                    input_sub = self.input[i, x*self.strides:(x*self.strides)+self.pool_size, y*self.strides:(y*self.strides)+self.pool_size]
                    max = np.max(input_sub)
                    result = unravel_index(input_sub.argmax(), input_sub.shape)
                    max_x = result[0]
                    max_y = result[1]
                    input_gradient[i, x*self.strides:(x*self.strides)+self.pool_size, y*self.strides:(y*self.strides)+self.pool_size][max_x, max_y] = max
        
        return input_gradient

class Flatten(Layer):

    def __init__(self, input_size=None):
        super().__init__(None)
        self.name = "Flatten"
        self.input_size = input_size
        self.output_size = (2*((input_size[1]**2))) #Temporary 

    def init_weights(self, input_size):
        pass

    def forward_propagation(self, input):

        #print("Flatten Input Shape: " + str(input.shape))

        self.before_flattened_shape = input.shape
        output = np.array([input.flatten()])

        return output

    def backward_propagation(self, output_error, learning_rate):
        before_flattened = output_error.reshape(self.before_flattened_shape)
        return before_flattened

