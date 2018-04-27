import numpy as np


class Linear:

    def __init__(self, network, input_size, output_size, initial_weights=np.ndarray(0), initial_biases=np.ndarray(0)):
        """
        :param network: The neural network this layer belongs to.
        :param input_size: Amount of input neurons.
        :param output_size: Amount of output neurons.
        :param initial_weights: Initial weights. Random ones are seeded if none given. Giving initial weights makes
        teh functionality of the layer deterministic.
        :param initial_biases: Initial biases for this layer. Random ones are seeded if none given.
        """
        self.network = network
        if input_size < 0 or output_size < 0:
            raise ValueError('Input and output sizes have to be positive for linear layer')
        if initial_weights.size > 0 and not initial_weights.shape == (output_size, input_size):
            raise ValueError('Initial weights not the right dimension: {} not {}'
                             .format(initial_weights, (output_size, input_size)))
        if initial_biases.size > 0 and not initial_biases.shape == (output_size, ):
            raise ValueError('Initial biases not the right dimension: {} not {}'
                             .format(initial_biases.shape, (output_size,)))

        self.weights = initial_weights if initial_weights.size > 0 else np.random.random(size=(output_size, input_size))
        self.biases = initial_biases if initial_biases.size > 0 else np.ones(output_size)

    def forward(self, input):
        """
        This is basically a wrapper for forward_pass, where we save all the parameters needed in backpropagation
        :param input: input for a specific layer of shape (batch_size, input_dimension)
        :return: output of this layer after adding weights and biases of shape (batch_size, output_dimension)
        """
        # so if it's not the first forward pass let's take the updated weights
        if len(self.network.saved_weights) > 0 and hasattr(self, 'index'):
            self.weights = self.network.saved_weights[self.index]
        if len(self.network.saved_biases) > 0 and hasattr(self, 'index'):
            self.biases = self.network.saved_biases[self.index]
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
        """
        Where the matrix magic happens. Sum the products of the outputs of each neuron of the last layer with its weight.
        Add biases in the end.
        :param input: input for a specific layer of shape (batch_size, input_dimension)
        :return: output of this layer after adding weights and biases of shape (batch_size, output_dimension)
        """
        return np.dot(input, self.weights.T) + self.biases


