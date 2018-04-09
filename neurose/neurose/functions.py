import numpy as np
from functools import reduce
from functools import reduce


# Just your average activation function
class Sigmoid():

    @staticmethod
    def call(x):
        return 1/(1 + np.exp(-np.array(x)))

    @staticmethod
    def derivative(x):
        s = np.array(Sigmoid.call(x))
        return s*(1-s)


# Maps values in an array between [0,1]
# Is used mostly in output layer to obtain probabilities
class SoftMax():

    @staticmethod
    def call(inp):
        # All this transposing just to support batches
        x = inp.T
        out = []
        for i in range(len(x)):
            z = reduce(lambda val, sum: sum + val, np.exp(x[i]))
            out.append(np.exp(x[i])/z)
        return np.asarray(out).T

    @staticmethod
    def derivative(x):
        s = np.array(SoftMax.call(x))
        return s*(1-s)

# Just your average loss function
class MeanSquaredError():

    # We assume that the input is a matrix with each row representing a different output
    # so the matrix contains all the outputs of a single batch
    @staticmethod
    def call(outputs, labels):
        if not isinstance(outputs, np.ndarray) or not isinstance(labels, np.ndarray):
            raise ValueError('Loss functions require lists as inputs: {}'.format(outputs, labels))
        if not len(outputs) == len(labels):
            raise ValueError('Outputs and labels are a different length: {} and {}'.format(len(outputs), len(labels)))
        sum = 0
        for o, l in zip(outputs, labels):
            if not len(o) == len(l): raise ValueError('Outputs and labels are a different dimesion: {} and {}'.format(o, l))
            # Use Euclidean distance between vectors to define a pass specific error
            sum += (np.linalg.norm(o - l)) ** 2
        return sum/(len(outputs)*len(outputs[0]))

    @staticmethod
    def derivative(x):
        raise NotImplementedError