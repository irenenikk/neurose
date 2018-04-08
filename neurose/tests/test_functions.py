from unittest import TestCase
from torch.nn import Sigmoid as TorchSigmoid
from torch.nn import MSELoss as TorchMSE
from torch.autograd import Variable
import torch
from random import randint
from neurose.functions import Sigmoid, SoftMax, MeanSquaredError
import numpy as np


class TestFunctions(TestCase):

    def test_sigmoid(self):
        rand = [[randint(-10, 10) for i in range(50)] for i in range(30)]
        torch_func = TorchSigmoid()
        torch_result = torch_func(torch.DoubleTensor(rand))
        self.assertTrue((torch_result == torch.from_numpy(Sigmoid.call(rand)).double()).all())

    def test_softmax(self):
        rand = np.asarray([[randint(0, 10) for i in range(2)] for i in range(3)])
        torch_result = torch.nn.functional.softmax(Variable(torch.DoubleTensor(rand)), dim=0)
        t = torch_result.data
        i = torch.from_numpy(SoftMax.call(rand))
        for x, y in zip(t, i):
            for i, j in zip(x, y):
                assert round(i, 10) == round(j, 10)

    def test_mean_squared_error(self):
        # torch's mean squared error uses batches dimensions differently so we can't directly test with it
        # we just test that the sum is correct
        a = 3
        b = 5
        outputs = np.asarray([[randint(0, 10) for j in range(a)] for i in range(b)])
        labels = np.asarray([[randint(0, 10) for j in range(a)] for i in range(b)])
        torch_func = TorchMSE(size_average=False)
        torch_result = torch_func(Variable(torch.from_numpy(outputs).double()), Variable(torch.from_numpy(labels).double()))
        my_result = MeanSquaredError.call(outputs, labels)
        assert (round(torch_result.data[0], 10) == round(my_result*len(outputs), 10))

    def test_mse_raises_error_on_incompatible_dimensions(self):
        a = np.asarray([[randint(0, 10) for j in range(2)] for i in range(5)])
        b = np.asarray([[randint(0, 10) for j in range(3)] for i in range(5)])
        self.assertRaises(ValueError, MeanSquaredError.call, a, b)

    def test_mse_raises_error_on_incompatible_lengths(self):
        a = np.asarray([[randint(0, 10) for j in range(2)] for i in range(5)])
        b = np.asarray([[randint(0, 10) for j in range(2)] for i in range(3)])
        self.assertRaises(ValueError, MeanSquaredError.call, a, b)
