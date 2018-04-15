import numpy as np
from neurose.net import Net
from neurose.layers import Linear
from neurose.functions import ReLu as s
from neurose.functions import SoftMax as p
from neurose.functions import MeanSquaredError as MSE


class Ex(Net):

    def __init__(self):
        super().__init__(MSE)
        # This is where the layers are defined
        self.a1 = s(self)
        self.a2 = p(self)
        #self.l1 = Linear(self, 4, 3)
        #self.l2 = Linear(self, 3, 3)
        #self.l3 = Linear(self, 3, 2)
        self.l1 = Linear(self, 2, 3)
        self.l2 = Linear(self, 3, 2)

    def forward_pass(self, input):
        # this is where forward pass is defined
        # you define which activation functions to use on which layer
        # and in which order will the layers be traversed
        # always passing the input forward
        # we only support one activation function per layer
        x = self.a1.call(self.l1.forward(input))
        x = self.a2.call(self.l2.forward(x))
        #x = self.a2.call(self.l3.forward(x))
        return x


e = Ex()

# the input consists of three batches and inputs of dimension 4
for i in range(500):
    e.reset_saved_parameters()

    xor_input = [[1, 1], [1, 0], [0, 1], [0, 0]]
    output = e.forward(xor_input)

    xor_labels = [[1, 0], [0, 1], [0, 1], [1, 0]]
    actual = np.asarray(xor_labels)

    loss = e.calculate_loss(output, actual)

    print('loss: {}'.format(loss))

    e.backpropagate()

    e.update_weights()

    print('out: {}: '.format(output))