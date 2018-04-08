from neurose.matrix_math import MatrixMath
import numpy as np


class Layer():

    def __init__(self, network):
        self.network = network

    def forward_pass(self, input):
        self.network.save_input(input)
        output = self.forward(input)
        self.network.save_output(output)
        return output

    def forward(self):
        raise NotImplementedError


class Linear(Layer):

    def __init__(self, network, input_size, output_size):
        self.network = network
        self.weights = np.random.random((output_size, input_size))
        self.biases = np.random.random((output_size, 1))

    def forward(self, input):
        return np.dot(self.weights, input) + self.biases

