import numpy as np


class Net:

    def __init__(self, loss_function, learning_rate=0.5):
        self.loss = loss_function
        self.learning_rate = learning_rate
        self.init_parameters()
        self.saved_weights = []
        self.saved_biases = []
        self.saved_activation_functions = []

    def init_parameters(self):
        """
        Reset the parameters which change on each forward pass.
        """
        self.saved_inputs = []
        self.saved_outputs = []

    def save_input(self, input):
        self.saved_inputs.append(input)

    def save_output(self, output):
        self.saved_outputs.append(output)

    def save_weights_and_biases(self, weights, biases, index=None):
        """
        Save the weights and biases of a specific layer.
        :param index: Enables retrieving weights which were updated in backpropagation.
        :return: The index of the weights and biases. Returned on the first forward pass.
        """
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
        """
        This is just to make sure that a numpy array is passed forward
        """
        return self.forward_pass(np.array(input))

    def forward_pass(self, x):
        """
        This is where forward pass is defined.
        You define which activation functions to use on which layer
        and in which order will the layers be traversed, always passing the input forward.
        Note that we only support one activation function per layer
        """
        raise NotImplementedError

    def calculate_loss(self, predictions, labels):
        """
        Calculate the loss of a specific batch i.e. how wrong were we.
        :param predictions: Output of the neural network of shape (batch_size, output_dimension)
        :param labels: True labels of each input of shape (batch_size, output_dimension)
        """
        if not predictions.shape == labels.shape:
            raise ValueError('predictions and labels not the same shape: {} and {}'.format(predictions.shape, labels.shape))
        self.loss_derivative = self.loss.derivative(predictions, labels)
        return self.loss.call(predictions, labels)

    def backpropagate(self):
        """
        Calculate how much each weight affected the loss of the forward pass.
        This is done using derivatives. See notes about backpropagation in the wiki for some neat math stuff.
        """
        if not hasattr(self, 'loss_derivative'):
            raise ValueError('calculate_loss not called before backpropagation')
        no_layers = len(self.saved_weights) + 1
        errors = [None] * no_layers
        # base case error
        # loss_derivative is defined when loss is calculated
        errors[-1] = self.loss_derivative * self.saved_activation_functions[-1].derivative(self.saved_inputs[-1])
        # the last layer was the base case, so we don't iterate over it
        # range is not inclusive of the ending point
        for i in range(no_layers - 2, -1, -1):
            errors[i] = np.dot(errors[i+1], np.asarray(self.saved_weights[i])) \
                       * self.saved_activation_functions[i].derivative(self.saved_inputs[i])
        # the array now holds the errors of each layer
        self.errors = errors

    def update_weights(self):
        if not hasattr(self, 'errors'):
            raise ValueError('backpropagate not called before updating weights')
        # the gradients are used in tests
        gradients = []
        for i in range(len(self.saved_weights)):
            gradient = np.dot(self.errors[i+1].T, self.saved_outputs[i])
            self.saved_weights[i] -= gradient * self.learning_rate
            gradients.append(gradient)
        return gradients

    def reset_saved_parameters(self):
        """
        Reset the saved inputs and outputs used in backpropagation.
        """
        self.init_parameters()