import numpy as np
from net import Net
from layers import Linear
from functions import ReLu, SoftMax, Passive
from functions import MeanSquaredError as MSE


class Ex(Net):

    def __init__(self):
        super().__init__(MSE, learning_rate=0.02)
        # This is where the layers are defined
        self.a1 = Passive(self)
        self.l1 = Linear(self, 1, 5)
        self.l2 = Linear(self, 5, 5)
        self.l3 = Linear(self, 5, 1)

    def forward_pass(self, input):
        # this is where forward pass is defined
        # you define which activation functions to use on which layer
        # and in which order will the layers be traversed
        # always passing the input forward
        # we only support one activation function per layer
        x = self.a1.call(self.l1.forward(input))
        x = self.a1.call(self.l2.forward(x))
        x = self.a1.call(self.l3.forward(x))
        return x


e = Ex()

# the input consists of three batches and inputs of dimension 4
for i in range(100):
    e.reset_saved_parameters()

    input = np.asarray([[1], [2], [3], [4]])
    output = e.forward(input)

    label = []
    for i in input:
        for j in i:
            label.append([2*j])
    actual = np.asarray(label)

    loss = e.calculate_loss(output, actual)

    # print('loss: {}'.format(loss))

    e.backpropagate()

    e.update_weights()

    print('out: {}: '.format(output))