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
        self.l2 = Linear(self, 5, 1)

    def forward_pass(self, input):
        # this is where forward pass is defined
        # you define which activation functions to use on which layer
        # and in which order will the layers be traversed
        # always passing the input forward
        # we only support one activation function per layer
        x = self.a1.call(self.l1.forward(input))
        x = self.a1.call(self.l2.forward(x))
        return x


e = Ex()

# here we are trying to teach the neural network a simple linear regression problem
# for now this is just overfitting to the same few data points to show that the network can actually learn something
# look at the output: it should converge to [2, 4, 6, 8]
# there's still a weird bug where sometimes the network goes to a completely wrong direction and ends up printing inf/nan
for i in range(200):
    e.reset_saved_parameters()

    input = np.asarray([[1], [2], [3], [4]])
    output = e.forward(input)

    label = []
    for i in input:
        for j in i:
            label.append([2*j])
    actual = np.asarray(label)

    loss = e.calculate_loss(output, actual)

    print('loss: {}'.format(loss))

    e.backpropagate()

    e.update_weights()

    print('out: \n{}: '.format(output))