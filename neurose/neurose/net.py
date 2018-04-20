import numpy as np


class Net:

    def __init__(self, loss_function, learning_rate=0.5):
        self.loss = loss_function
        self.learning_rate = learning_rate
        self.init_parameters()
        self.saved_weights = []
        self.saved_biases = []

    def init_parameters(self):
        self.saved_inputs = []
        self.saved_outputs = []
        self.saved_activation_functions = []

    def save_input(self, input):
        self.saved_inputs.append(input)

    def save_output(self, output):
        self.saved_outputs.append(output)

    def save_weights_and_biases(self, weights, biases, index=None):
        if index is not None:
            self.saved_weights[index] = weights
            self.saved_biases[index] = biases
        else:
            self.saved_weights.append(weights)
            self.saved_biases.append(biases)
            return len(self.saved_weights) - 1

    def save_activation_function(self, func):
        self.saved_activation_functions.append(func)

    def forward(self, input):
        # The transposing needs to be done for the matrix multiplication to work
        return self.forward_pass(np.array(input).T).T

    def forward_pass(self, x):
        # This is where forward pass is defined
        # You define which activation functions to use on which layer
        # and in which order will the layers be traversed
        raise NotImplementedError

    def calculate_loss(self, predictions, labels):
        if not predictions.shape == labels.shape:
            raise ValueError('predictions and labels not the same shape: {} and {}'.format(predictions.shape, labels.shape))
        self.loss_derivative = self.loss.derivative(predictions, labels)
        return self.loss.call(predictions, labels)

    def backpropagate(self):
        print('backpropagating')
        if not hasattr(self, 'loss_derivative'):
            raise ValueError('calculate_loss not called before backpropagation')
        no_layers = len(self.saved_weights) + 1
        errors = [None] * no_layers
        # base case error
        errors[-1] = self.loss_derivative * self.saved_activation_functions[-1].derivative(self.saved_inputs[-1])
        for i in range(no_layers - 2, -1, -1):
            errors[i] = np.dot(np.asarray(self.saved_weights[i].T), errors[i+1]) \
                       * self.saved_activation_functions[i].derivative(self.saved_inputs[i])
        self.errors = errors

    def update_weights(self):
        if not hasattr(self, 'errors'):
            raise ValueError('backpropagate not called before updating weights')
        # output layer is not backpropagated
        gradients = []
        for i in range(len(self.saved_weights)):
            gradient = np.dot(self.errors[i+1], self.saved_inputs[i].T)
            self.saved_weights[i] -= gradient * self.learning_rate
            gradients.append(gradient)
        return gradients

    def reset_saved_parameters(self):
        self.init_parameters()