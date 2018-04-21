import numpy as np
from functools import reduce
import math


# parent class for all activation functions
# is used to store the activation functions of each layer for backpropagation
class DifferentiableFunction:

    def __init__(self, net):
        self.net = net

    def call(self, x):
        # for backpropagation
        self.net.save_activation_function(self)
        output = self.func(x)
        # for backpropagation
        self.net.save_output(output)
        return output

    def func(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

# When we don't want to use any activation
class Passive(DifferentiableFunction):

    def func(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)


# Just your average activation function
class Sigmoid(DifferentiableFunction):

    def func(self, x):
        return 1/(1 + np.exp(-np.array(x)))

    def derivative(self, x):
        s = np.array(self.func(x))
        return s*(1-s)


# Maps values in an array between [0,1]
# Is used mostly in output layer to obtain probabilities
class SoftMax(DifferentiableFunction):

    def func(self, inp):
        # All this transposing just to support batches
        x = inp.T
        out = []
        for i in range(len(x)):
            z = reduce(lambda val, sum: sum + val, np.exp(x[i]))
            out.append(np.exp(x[i])/z)
        return np.asarray(out).T

    def derivative(self, x):
        s = np.array(self.call(x))
        return s*(1-s)


# Just your average activation function
class ReLu(DifferentiableFunction):

    def func(self, x):
        return np.maximum(x, 0)

    def derivative(self, x):
        for i, val1 in enumerate(x):
            for j, val2 in enumerate(val1):
                val = x[i][j]
                x[i][j] = 1 if val > 0 else 0
        return x

# Just your average loss function
class MeanSquaredError:

    # We assume that the input is a matrix with each row representing a different output
    # So the matrix contains all the outputs of a single batch
    @staticmethod
    def call(outputs, labels):
        if not isinstance(outputs, np.ndarray) or not isinstance(labels, np.ndarray):
            raise ValueError('Loss functions require lists as inputs: {}'.format(outputs, labels))
        if not len(outputs) == len(labels):
            raise ValueError('Outputs and labels are a different length: {} and {}'.format(len(outputs), len(labels)))
        if len(outputs) == 0 or len(labels) == 0:
            raise ValueError('No outputs or labels given')
        sum = 0
        for o, l in zip(outputs, labels):
            if not len(o) == len(l): raise ValueError('Outputs and labels are of different dimesion: {} and {}'.format(o, l))
            sum += (o - l) ** 2
        # The mean is calculated element wise
        mean = sum/(len(outputs)*len(outputs[0]))
        return mean

    # calculations in notes about backpropagation
    @staticmethod
    def derivative(outputs, labels):
        batch_size = len(outputs)
        dimension = len(outputs[0])
        return 2/(batch_size*dimension)*(outputs - labels)

