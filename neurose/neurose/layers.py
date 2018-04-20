import numpy as np


class Linear:

    def __init__(self, network, input_size, output_size, initial_weights=np.ndarray(0), initial_biases=np.ndarray(0)):
        self.network = network
        if input_size < 0 or output_size < 0:
            raise ValueError('Input and output sizes have to be positive for linear layer')
        if initial_weights.size > 0 and not initial_weights.shape == (output_size, input_size):
            raise ValueError('Initial weights not the right dimension: {} not {}'
                             .format(initial_weights, (output_size, input_size)))
        if initial_biases.size > 0 and not initial_biases.shape == (output_size,1):
            raise ValueError('Initial biases not the right dimension: {} not {}'
                             .format(initial_biases.shape, (output_size,1)))
        # initialize random weights if none given
        self.weights = initial_weights if initial_weights.size > 0 else np.random.normal(size=(output_size, input_size))
        self.biases = initial_biases if initial_biases.size > 0 else np.random.random((output_size, 1))

    # this is basically a wrapper for forward_pass
    # where we save all the parameters needed in backpropagation
    def forward(self, input):
        # the weights have to be updated, which is done by the network
        # so if it's not the first forward pass let's take the updated weights
        if len(self.network.saved_weights) > 0 and hasattr(self, 'index'):
            self.weights = self.network.saved_weights[self.index]
        if len(self.network.saved_inputs) == 0 and len(self.network.saved_outputs) == 0:
            # the first layer just passes the input to hidden layers
            # these are used in backpropagation
            self.network.save_input(input)
            self.network.save_output(input)
        # actual forward pass
        weighted = self.forward_pass(input)
        # for backpropagation
        self.network.save_input(weighted)
        if self.weights is None or self.biases is None:
            raise ValueError('No weights or biases defined for layer: {}'.format(self))
        # index tells where we can get the updated weights after backpropagation
        if hasattr(self, 'index'):
            # update old weights
            self.network.save_weights_and_biases(self.weights, self.biases, self.index)
        else:
            # first time passing: add weights and acquire index so you can keep updating the weights
            self.index = self.network.save_weights_and_biases(self.weights, self.biases)
        return weighted

    def forward_pass(self, input):
        return np.dot(self.weights, input) + self.biases


