import numpy as np
from net import Net
from layers import Linear
from functions import ReLu, CrossEntropy, SoftMax
import torchvision
import torch


class Ex(Net):

    def __init__(self):
        super().__init__(CrossEntropy, learning_rate=0.02)
        """This is where the layers are defined"""
        self.a1 = ReLu(self)
        self.a2 = SoftMax(self)
        self.l1 = Linear(self, 784, 256)
        self.l2 = Linear(self, 256, 120)
        self.l3 = Linear(self, 120, 64)
        self.l4 = Linear(self, 64, 10)

    def forward_pass(self, input):
        x = input.reshape(input.shape[0], -1)
        x = self.a1.call(self.l1.forward(x))
        x = self.a1.call(self.l2.forward(x))
        x = self.a1.call(self.l3.forward(x))
        x = self.a2.call(self.l4.forward(x))
        return x


e = Ex()

"""
See notes about training a neural network in the wiki.
"""

dataset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=256, num_workers=2)

for epoch in range(500):

    for x, y in dataloader:

        e.reset_saved_parameters()

        input = x.numpy()
        input = input.astype(np.float128)
        actual = y.numpy()

        output = e.forward_pass(input)

        print(output)

        loss = e.calculate_loss(output, actual)

        print('loss for expoch {}: {}'.format(epoch, loss))

        e.backpropagate()

        e.update_weights()

    print('out: \n{}: '.format(output))
