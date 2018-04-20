from unittest import TestCase
import numpy as np
from random import randint
from neurose.layers import Linear
from neurose.net import Net
from neurose.functions import Passive
from neurose.functions import MeanSquaredError as MSE
import torch
from torch.autograd import Variable
from torch import nn, optim
import json

weights1 = [[1], [2], [3], [4], [5]]
weights2 = [[6, 7, 8, 9, 10]]
my_biases1 = [[1], [1], [1], [1], [1]]
my_biases2 = [[1]]
torch_biases1 = [1, 1, 1, 1, 1]
torch_biases2 = [1]

class Test(Net):

    def __init__(self):
        super().__init__(MSE, learning_rate=0.02)
        # This is where the layers are defined
        self.a1 = Passive(self)
        self.l1 = Linear(self, 1, 5, np.asarray(weights1), np.asarray(my_biases1))
        self.l3 = Linear(self, 5, 1, np.asarray(weights2), np.asarray(my_biases2))

    def forward_pass(self, input):
        # this is where forward pass is defined
        # you define which activation functions to use on which layer
        # and in which order will the layers be traversed
        # always passing the input forward
        # we only support one activation function per layer
        x = self.a1.call(self.l1.forward(input))
        x = self.a1.call(self.l3.forward(x))
        return x


class TestNet(TestCase):

    def test_calculate_loss_raises_error_if_wrong_dimensions(self):
        pred = np.asarray([[i for i in range (2)] for j in range(3)])
        lab = np.asarray([[i for i in range (3)] for j in range(2)])
        n = Net(MSE)
        self.assertRaises(ValueError, n.calculate_loss, pred, lab)

    def test_calculate_loss_sets_final_output_derivative(self):
        pred = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        lab = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        n = Net(MSE)
        n.calculate_loss(pred, lab)
        assert not n.loss_derivative == 0

    def test_backpropagation_raises_error_if_calculate_loss_not_called(self):
        pred = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        lab = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        n = Net(MSE)
        self.assertRaises(ValueError, n.backpropagate)

    def test_update_weights_raises_error_if_backpropagate_not_called(self):
        pred = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        lab = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        n = Net(MSE)
        self.assertRaises(ValueError, n.update_weights)

    def test_forward_pass(self):
        # this is not ready yet
        global weights1
        global weights2
        global biases1
        global biases2
        # see that a forward pass results in the same parameters
        # give the same initial weights
        torch_network = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 1)
        )
        for i, param in enumerate(torch_network.parameters()):
            if i == 0:
                param.data = torch.FloatTensor(weights1)
            elif i == 1:
                param.data = torch.FloatTensor(torch_biases1)
            elif i == 2:
                param.data = torch.FloatTensor(weights2)
            elif i == 3:
                param.data = torch.FloatTensor(torch_biases2)

        e = Test()
        input = np.asarray([[1], [2], [3], [4]])
        x = Variable(torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1))
        output = e.forward(input)
        label = []
        for i in input:
            for j in i:
                label.append([2 * j])
        actual = np.asarray(label)
        my_loss = e.calculate_loss(output, actual)
        y = Variable(x.data * 2)
        y_pred = torch_network(x)
        loss_function = nn.MSELoss()
        torch_loss = loss_function(y_pred, y)
        torch_loss.backward()
        e.backpropagate()
        for i in torch_network.parameters():
            print('torch parameters: \n{}'.format(i))
        for i in e.saved_weights:
            print('my weights: {}'.format(i))

    # def test_backpropagate(self):
    #     # create torch net with these weights
    #     torch.manual_seed(10)
    #     np.random.seed(10)
    #     torch_network = nn.Sequential(
    #         nn.Linear(1, 5),
    #         nn.Linear(5, 1)
    #     )
    #     loss_function = nn.MSELoss()
    #     x = Variable(torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1))
    #     y = Variable(x.data * 2)
    #     y_pred = torch_network(x)
    #     for i in torch_network.parameters():
    #         print('torch parameters: \n{}'.format(i))
    #     torch_loss = loss_function(y_pred, y)
    #     torch_loss.backward()
    #     print(torch_network)
    #     print('torch loss: {}'.format(torch_loss))
    #
    #     # my network
    #     e = Test()
    #     input = np.asarray([[1], [2], [3], [4]])
    #     output = e.forward(input)
    #     label = []
    #     for i in input:
    #         for j in i:
    #             label.append([2 * j])
    #     actual = np.asarray(label)
    #     my_loss = e.calculate_loss(output, actual)
    #     print('my loss: {}'.format(my_loss))
    #     e.backpropagate()
    #     gradients = e.update_weights()
    #     print('my weights: \n{}'.format(e.saved_weights))
    #     print('my gradients: \n{}'.format(gradients))
