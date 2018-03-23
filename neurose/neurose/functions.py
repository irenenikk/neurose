import numpy as np
import math


def sigmoid(x):
    return 1/(1 + np.exp(-np.array(x)))


def d_sigmoid(x):
    s = np.array(sigmoid(x))
    return s*(1-s)

