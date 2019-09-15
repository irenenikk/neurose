from unittest import TestCase
import numpy as np
from random import randint
from neurose.layers.linear import Linear
from neurose.net import Net
from neurose.functions import Passive, Sigmoid, ReLu, SoftMax
from neurose.functions import MeanSquaredError as MSE
from neurose.functions import CrossEntropy as Cross
import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F

weights1 = [[1.], [2.], [3.], [4.], [5.]]
weights2 = [[6., 7., 8., 9., 10.]]
xor_weights1 = [[1., 1], [2., 2], [3., 3], [4., 4.], [5., 5.]]
xor_weights2 = [[6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.]]
my_biases1 = [1, 1, 1, 1, 1]
my_biases2 = [1]
xor_biases1 = [1, 1, 1, 1, 1]
xor_biases2 = [1, 1]


class NetWithoutForward(Net):
    def __init__(self):
        super().__init__(MSE, learning_rate=0.02)


class SimpleTest(Net):

    def __init__(self):
        super().__init__(MSE, learning_rate=0.02)
        self.a1 = Passive(self)
        self.l1 = Linear(self, 1, 5, np.asarray(weights1), np.asarray(my_biases1))
        self.l3 = Linear(self, 5, 1, np.asarray(weights2), np.asarray(my_biases2))

    def forward_pass(self, input):
        x = self.a1.call(self.l1.forward(input))
        x = self.a1.call(self.l3.forward(x))
        return x


class WithActivation(Net):

    def __init__(self):
        super().__init__(MSE, learning_rate=0.02)
        self.a1 = ReLu(self)
        self.a2 = Sigmoid(self)
        self.l1 = Linear(self, 1, 5, np.asarray(weights1), np.asarray(my_biases1))
        self.l3 = Linear(self, 5, 1, np.asarray(weights2), np.asarray(my_biases2))

    def forward_pass(self, input):
        x = self.a1.call(self.l1.forward(input))
        x = self.a2.call(self.l3.forward(x))
        return x


class TorchWithActivation(nn.Module):

    def __init__(self):
        super(TorchWithActivation, self).__init__()
        self.l1 = nn.Linear(1, 5)
        self.l2 = nn.Linear(1, 5)

    def forward(self, x):
       x = F.relu(self.l1(x))
       return torch.sigmoid(self.l2(x))


class WithSoftmax(Net):

    def __init__(self):
        super().__init__(Cross, learning_rate=0.02)
        self.a1 = Passive(self)
        self.a2 = SoftMax(self)
        self.l1 = Linear(self, 2, 5, np.asarray(xor_weights1), np.asarray(xor_biases1))
        self.l3 = Linear(self, 5, 2, np.asarray(xor_weights2), np.asarray(xor_biases2))

    def forward_pass(self, input):
        x = self.a1.call(self.l1.forward(input))
        x = self.a2.call(self.l3.forward(x))
        return x


class TorchWithSoftmax(nn.Module):

    def __init__(self):
        super(TorchWithSoftmax, self).__init__()
        self.l1 = nn.Linear(2, 5)
        self.l2 = nn.Linear(5, 2)

    def forward(self, x):
       x = self.l1(x)
       return F.softmax(self.l2(x), dim=1)


class TestNet(TestCase):

    def test_forward_pass_defined_by_use(self):
        net = NetWithoutForward()
        self.assertRaises(NotImplementedError, net.forward, [1])

    def test_calculate_loss_raises_error_if_wrong_dimensions(self):
        pred = np.asarray([[i for i in range (2)] for j in range(3)])
        lab = np.asarray([[i for i in range (3)] for j in range(2)])
        n = Net(MSE)
        self.assertRaises(ValueError, n.calculate_loss, pred, lab)

    def test_calculate_loss_sets_final_output_derivative(self):
         pred = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
         # make sure there is loss
         lab = pred + 1
         n = Net(MSE)
         n.calculate_loss(pred, lab)
         assert not (n.loss_derivative == 0).any()

    def test_calculate_loss_sets_final_output_derivative_on_success(self):
         pred = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
         # make sure there is no loss
         lab = pred
         n = Net(MSE)
         n.calculate_loss(pred, lab)
         assert (n.loss_derivative == 0).all()

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


    def test_forward_pass_no_activation(self):
        # give the same initial weights
        # see that output is the same
        # train torch
        torch_network = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 1),
        )
        opt = optim.SGD(torch_network.parameters(), lr=0.02)
        self.set_initial_weights(torch_network)
        # train neurose
        e = SimpleTest()
        input = self.get_lin_regression_inputs()
        x = Variable(torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1))
        # compare outputs
        my_output = e.forward(input)
        torch_output = torch_network(x)
        self.assert_list_is_equal_to_tensor(my_output, torch_output)

    def test_weight_backpropagation_no_activation(self):
        # give the same initial weights
        # check that weights and gradients are the same after backpropagation
        # train neurose network
        e = SimpleTest()
        gradients = self.do_neurose_training_round_with_mse(e)
        # train torch network
        torch_network = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 1),
        )
        self.do_torch_training_round_with_mse(torch_network)
        torch_biases, torch_weights = self.collect_torch_variables(torch_network)
        for t, m in zip(torch_weights, e.saved_weights):
            self.assert_list_is_equal_to_tensor(m, t)
        for t, m in zip(torch_weights, gradients):
            self.assert_list_is_equal_to_tensor(m, t.grad)
        for t, m in zip(torch_biases, e.saved_biases):
            self.assert_vector_is_equal_to_tensor(m, t)

    def test_forward_pass_with_activation(self):
        # with torch
        torch_network = TorchWithActivation()
        opt = optim.SGD(torch_network.parameters(), lr=0.02)
        self.set_initial_weights(torch_network)
        # with neurose
        e = WithActivation()
        input = self.get_lin_regression_inputs()
        x = Variable(torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1))
        # check that outuputs are the same
        my_output = e.forward(input)
        torch_output = torch_network(x)
        self.assert_list_is_equal_to_tensor(my_output, torch_output)

    def test_weight_backpropagation_with_activation(self):
        # train neurose
        e = WithActivation()
        gradients = self.do_neurose_training_round_with_mse(e)
        # train torch
        torch_network = TorchWithActivation()
        self.do_torch_training_round_with_mse(torch_network)
        torch_biases, torch_weights = self.collect_torch_variables(torch_network)
        for t, m in zip(torch_weights, e.saved_weights):
            self.assert_list_is_equal_to_tensor(m, t)
        for t, m in zip(torch_weights, gradients):
            self.assert_list_is_equal_to_tensor(m, t.grad)
        for t, m in zip(torch_biases, e.saved_biases):
            self.assert_vector_is_equal_to_tensor(m, t)

    def test_forward_pass_with_softmax(self):
        # with torch
        torch_network = TorchWithSoftmax()
        opt = optim.SGD(torch_network.parameters(), lr=0.02)
        self.set_xor_initial_weights(torch_network)
        # with neurose
        e = WithSoftmax()
        input = self.get_xor_input()
        x = Variable(torch.FloatTensor([[1, 0], [0, 0], [0, 1], [1, 1]]))
        # check that outuputs are the same
        my_output = e.forward(input)
        torch_output = torch_network(x)
        self.assert_list_is_equal_to_tensor(my_output, torch_output)

    def test_weight_backpropagation_with_softmax(self):
        # train neurose
        e = WithSoftmax()
        gradients = self.do_neurose_training_round_with_cross(e)
        # train torch
        torch_network = TorchWithSoftmax()
        self.do_torch_training_round_with_cross(torch_network)
        torch_biases, torch_weights = self.collect_torch_variables(torch_network)
        for t, m in zip(torch_weights, e.saved_weights):
            self.assert_list_is_equal_to_tensor(m, t)
        for t, m in zip(torch_weights, gradients):
            self.assert_list_is_equal_to_tensor(m, t.grad)
        for t, m in zip(torch_biases, e.saved_biases):
            self.assert_vector_is_equal_to_tensor(m, t)

    def assert_list_is_equal_to_tensor(self, my_output, torch_output):
        for my, torch_result in zip(my_output, torch_output.data):
            assert round(my[0], 1) == round(torch_result[0].item(), 1)

    def set_initial_weights(self, torch_network):
        global weights1
        global weights2
        global biases1
        global biases2
        for i, param in enumerate(torch_network.parameters()):
            if i == 0:
                param.data = torch.FloatTensor(weights1)
            elif i == 1:
                param.data = torch.FloatTensor(my_biases1)
            elif i == 2:
                param.data = torch.FloatTensor(weights2)
            elif i == 3:
                param.data = torch.FloatTensor(my_biases2)

    def set_xor_initial_weights(self, torch_network):
        global xor_weights1
        global xor_weights2
        global xor_biases1
        global xor_biases2
        for i, param in enumerate(torch_network.parameters()):
            if i == 0:
                param.data = torch.FloatTensor(xor_weights1)
            elif i == 1:
                param.data = torch.FloatTensor(xor_biases1)
            elif i == 2:
                param.data = torch.FloatTensor(xor_weights2)
            elif i == 3:
                param.data = torch.FloatTensor(xor_biases2)

    def collect_torch_variables(self, torch_network):
        torch_weights = []
        torch_biases = []
        for i, p in enumerate(torch_network.parameters()):
            if i % 2 == 0:
                torch_weights.append(p)
            else:
                torch_biases.append(p)
        return torch_biases, torch_weights

    def assert_vector_is_equal_to_tensor(self, m, t):
        for i in range(len(t)):
            assert round(m[i], 2) == round(t[i].item(), 2)

    def do_torch_training_round_with_mse(self, torch_network):
        opt = optim.SGD(torch_network.parameters(), lr=0.02)
        self.set_initial_weights(torch_network)
        x = Variable(torch.FloatTensor([1, 2, 3, 4]).unsqueeze(1))
        y = Variable(x.data * 2)
        y_pred = torch_network(x)
        loss_function = nn.MSELoss()
        torch_loss = loss_function(y_pred, y)
        torch_loss.backward()
        opt.step()

    def do_neurose_training_round_with_mse(self, e):
        input = self.get_lin_regression_inputs()
        output = e.forward(input)
        label = self.get_lin_regression_labels(input)
        actual = np.asarray(label)
        e.calculate_loss(output, actual)
        e.backpropagate()
        gradients = e.update_weights()
        return gradients

    def do_torch_training_round_with_cross(self, torch_network):
        opt = optim.SGD(torch_network.parameters(), lr=0.02)
        self.set_xor_initial_weights(torch_network)
        x = Variable(torch.FloatTensor([[1, 0], [0, 0], [0, 1], [1, 1]]))
        y = Variable(torch.LongTensor([1, 0, 1, 0]))
        y_pred = torch_network(x)
        loss_function = nn.CrossEntropyLoss()
        torch_loss = loss_function(y_pred, y)
        torch_loss.backward()
        opt.step()

    def do_neurose_training_round_with_cross(self, e):
        input = self.get_xor_input()
        output = e.forward(input)
        label = self.get_xor_labels(input)
        actual = np.asarray(label)
        e.calculate_loss(output, actual)
        e.backpropagate()
        gradients = e.update_weights()
        return gradients

    def get_lin_regression_inputs(self):
        return np.asarray([[1], [2], [3], [4]])

    def get_lin_regression_labels(self, input):
        label = []
        for i in input:
            for j in i:
                label.append([2 * j])
        return label

    def get_xor_input(self):
        return np.asarray([[1, 0], [0, 0], [0, 1], [1, 1]])

    def get_xor_labels(self, input):
        label = []
        for i in input:
            if i[0] == i[1]:
                label.append(1)
            else:
                label.append(0)
        return label


