from unittest import TestCase
from torch.nn import Sigmoid as TorchSigmoid
from torch.autograd import Variable
import torch
from random import randint
from neurose.functions import Sigmoid, SoftMax
import numpy as np


class TestFunctions(TestCase):

    def test_sigmoid(self):
        rand = [[randint(-10, 10) for i in range(50)] for i in range(30)]
        torch_func = TorchSigmoid()
        torch_result = torch_func(torch.DoubleTensor(rand))
        self.assertTrue((torch_result == torch.from_numpy(Sigmoid.call(rand)).double()).all())

    def test_softmax(self):
        a = 3
        b = 2
        rand = np.asarray([[randint(0, 10) for i in range(b)] for i in range(a)])
        torch_result = torch.nn.functional.softmax(Variable(torch.DoubleTensor(rand)), dim=0)
        t = torch_result.data
        i = torch.from_numpy(SoftMax.call(rand))
        for x, y in zip(t, i):
            for i, j in zip(x, y):
                assert round(i, 10) == round(j, 10)
