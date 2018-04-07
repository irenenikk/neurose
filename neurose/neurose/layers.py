from neurose.matrix_math import MatrixMath
import numpy as np


class Linear:

    def __init__(self, input_size, output_size):
        self.weights = np.random.random((output_size, input_size))
        self.biases = np.random.random((output_size, 1))

    def forward(self, input):
        return np.dot(self.weights, input) + self.biases

