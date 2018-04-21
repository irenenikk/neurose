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

weights1 = [[1.], [2.], [3.], [4.], [5.]]
weights2 = [[6., 7., 8., 9., 10.]]
my_biases1 = [1, 1, 1, 1, 1]
my_biases2 = [1]
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

    # def test_calculate_loss_sets_final_output_derivative(self):
    #     pred = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
    #     lab = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
    #     n = Net(MSE)
    #     n.calculate_loss(pred, lab)
    #     assert not n.loss_derivative == 0

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
        # give the same initial weights
        # see that output is the same
        torch_network = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 1)
        )
        opt = optim.SGD(torch_network.parameters(), lr=0.02)
        self.set_initial_weights(torch_network)
        e = Test()
        input = np.asarray([[1], [2], [3], [4]])
        x = Variable(torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1))
        my_output = e.forward(input)
        torch_output = torch_network(x)
        self.assert_list_is_equal_to_tensor(my_output, torch_output)

    def assert_list_is_equal_to_tensor(self, my_output, torch_output):
        for my, torch_result in zip(my_output, torch_output.data):
            assert round(my[0], 5) == round(torch_result[0], 5)

    def set_initial_weights(self, torch_network):
        global weights1
        global weights2
        global biases1
        global biases2
        for i, param in enumerate(torch_network.parameters()):
            if i == 0:
                param.data = torch.FloatTensor(weights1)
            elif i == 1:
                param.data = torch.FloatTensor(torch_biases1)
            elif i == 2:
                param.data = torch.FloatTensor(weights2)
            elif i == 3:
                param.data = torch.FloatTensor(torch_biases2)

    def test_backpropagation(self):
        # give the same initial weights
        # check that weights and gradients are the same after backpropagation
        torch_network = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 1)
        )
        opt = optim.SGD(torch_network.parameters(), lr=0.02)
        self.set_initial_weights(torch_network)
        e = Test()
        input = np.asarray([[1], [2], [3], [4]])
        x = Variable(torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1))
        output = e.forward(input)
        label = []
        for i in input:
            for j in i:
                label.append([2 * j])
        actual = np.asarray(label)
        e.calculate_loss(output, actual)
        y = Variable(x.data * 2)
        y_pred = torch_network(x)
        loss_function = nn.MSELoss()
        torch_loss = loss_function(y_pred, y)
        torch_loss.backward()
        e.backpropagate()
        gradients = e.update_weights()
        opt.step()
        # parameters() contains biases as well, so let's only take the weights
        torch_weights = []
        for i, p in enumerate(torch_network.parameters()):
            if i % 2 == 0:
                torch_weights.append(p)
        for t, m in zip(torch_weights, e.saved_weights):
            self.assert_list_is_equal_to_tensor(m, t)
        for t, m in zip(torch_weights, gradients):
            self.assert_list_is_equal_to_tensor(m, t.grad)

