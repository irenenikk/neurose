from neurose.functions import DifferentiableFunction
import numpy as np

class Net:

    def forward(self, input):
        # The transposing needs to be done for the matrix multiplication to work
        return self.forward_pass(np.array(input).T).T

    def forward_pass(self, x):
        # This is where forward pass is defined
        # You define which activation functions to use on which layer
        # and in which order will the layers be traversed
        raise NotImplementedError

    def backpropagate(self):
        pass

    def define_loss_function(self, loss_function):
        if not isinstance(loss_function, DifferentiableFunction):
            raise ValueError('Loss function should be a DifferentiableFunction')
        self.loss = loss_function

