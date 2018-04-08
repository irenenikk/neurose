import numpy as np
from functools import reduce


# Just your average activation function
class Sigmoid():

    def call(x):
        return 1/(1 + np.exp(-np.array(x)))

    def derivative(x):
        s = np.array(Sigmoid.call(x))
        return s*(1-s)


# Maps values in an array between [0,1]
# Is used mostly in output layer to obtain probabilities
class SoftMax():

    def call(inp):
        # All this transposing just to support batches
        x = inp.T
        out = []
        for i in range(len(x)):
            z = reduce(lambda val, sum: sum + val, np.exp(x[i]))
            out.append(np.exp(x[i])/z)
        return np.asarray(out).T

    def derivative(x):
        s = np.array(SoftMax.call(x))
        return s*(1-s)
