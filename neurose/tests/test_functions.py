from unittest import TestCase
from torch.nn import Sigmoid as TorchSigmoid
from torch.nn import MSELoss as TorchMSE
from torch.nn import CrossEntropyLoss as TorchCrossEntropy
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from random import randint
from neurose.functions import Sigmoid, SoftMax, MeanSquaredError, ReLu, CrossEntropy
from neurose.net import Net
import numpy as np
import random


class TestFunctions(TestCase):

    def test_sigmoid(self):
        rand = [[randint(-10, 10) for i in range(50)] for i in range(30)]
        torch_func = TorchSigmoid()
        my_func = Sigmoid(Net(MeanSquaredError))
        torch_result = torch_func(torch.DoubleTensor(rand))
        self.assertTrue((torch_result == torch.from_numpy(my_func.call(rand)).double()).all())

    def test_softmax(self):
        rand = np.asarray([[randint(0, 10) for i in range(2)] for i in range(3)])
        softmax = SoftMax(Net(MeanSquaredError))
        torch_result = F.softmax(Variable(torch.DoubleTensor(rand)), dim=1)
        t = torch_result.data
        i = torch.from_numpy(softmax.call(rand))
        for x, y in zip(t, i):
            for i, j in zip(x, y):
                assert round(i.item(), 10) == round(j.item(), 10)

    def test_relu(self):
        rand = [[randint(-10, 10) for i in range(50)] for i in range(30)]
        relu = ReLu(Net(MeanSquaredError))
        torch_result = F.relu(Variable(torch.DoubleTensor(rand)))
        t = torch_result.data
        i = torch.from_numpy(relu.call(rand))
        for x, y in zip(t, i):
            for i, j in zip(x, y):
                assert round(i.item(), 10) == round(j.item(), 10)


    def test_mean_squared_error(self):
        a = 3
        b = 5
        outputs = np.asarray([[randint(0, 10) for j in range(a)] for i in range(b)])
        labels = np.asarray([[randint(0, 10) for j in range(a)] for i in range(b)])
        torch_func = TorchMSE()
        torch_result = torch_func(Variable(torch.from_numpy(outputs).double()), Variable(torch.from_numpy(labels).double()))
        my_result = MeanSquaredError.call(outputs, labels)
        assert (round(torch_result.item(), 10) == round(my_result, 10))

    def test_mse_raises_error_on_incompatible_dimensions(self):
        a = np.asarray([[randint(0, 10) for j in range(2)] for i in range(5)])
        b = np.asarray([[randint(0, 10) for j in range(3)] for i in range(5)])
        self.assertRaises(ValueError, MeanSquaredError.call, a, b)

    def test_mse_raises_error_on_incompatible_lengths(self):
        a = np.asarray([[randint(0, 10) for j in range(2)] for i in range(5)])
        b = np.asarray([[randint(0, 10) for j in range(2)] for i in range(3)])
        self.assertRaises(ValueError, MeanSquaredError.call, a, b)

    def test_cross_entropy_loss(self):
        a = 2
        b = 3
        outputs = np.asarray([[random.uniform(0.1, 0.9) for j in range(a)] for i in range(b)])
        labels = np.asarray([randint(1, a-1) for i in range(b)])
        torch_func = TorchCrossEntropy()
        torch_result = torch_func(Variable(torch.from_numpy(outputs).double()), Variable(torch.from_numpy(labels).long()))
        my_result = CrossEntropy.call(outputs, labels)
        assert (round(torch_result.item(), 10) == round(my_result, 10))

