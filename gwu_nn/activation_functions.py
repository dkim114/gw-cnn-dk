import numpy as np
from abc import ABC, abstractmethod


def vectorize_activation(activation):
    def wrapper(*args):
        vec_activation = np.vectorize(activation)
        input = args[1]
        return vec_activation(input)
    return wrapper


class ActivationFunction(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def activation(cls, x):
        pass

    @abstractmethod
    def activation_partial_derivative(cls, x):
        pass


class SigmoidActivation(ActivationFunction):

    @classmethod
    def activation(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def activation_partial_derivative(cls, x):
        return np.exp(-x) / (1 + np.exp(-x))**2


class RELUActivation(ActivationFunction):

    @classmethod
    def activation(cls, x):
        vec = np.vectorize(cls.activation_func)
        return vec(x)

    @classmethod
    def activation_func(cls, x):
        if x > 0:
            return x
        else:
            return 0

    @classmethod
    def activation_partial_derivative(cls, x):
        vec = np.vectorize(cls.activation_parital_derivative_func)
        return vec(x)

    @classmethod
    def activation_parital_derivative_func(cls, x):
        if x > 0:
            return x
        else:
            return 0


class Softmax(ActivationFunction):

    @classmethod
    def activation(cls, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    @classmethod
    def activation_partial_derivative(cls, x):
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
