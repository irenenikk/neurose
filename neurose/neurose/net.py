import numpy as np

class Net:

    def __init__(self, loss_function, learning_rate=0.001):
        self.loss = loss_function
        self.learning_rate = learning_rate
        self.saved_inputs = []
        self.saved_outputs = []


    def save_input(self, input):
        # TODO
        self.saved_inputs.append(input)

    def save_output(self, output):
        # TODO
        self.saved_outputs.append(output)

    def forward(self, input):
        # The transposing needs to be done for the matrix multiplication to work
        output = self.forward_pass(np.array(input).T).T
        return output

    def forward_pass(self, x):
        # This is where forward pass is defined
        # You define which activation functions to use on which layer
        # and in which order will the layers be traversed
        raise NotImplementedError

    def backpropagate(self):
        pass


