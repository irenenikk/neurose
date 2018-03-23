from unittest import TestCase
from torch.nn import Sigmoid
import torch
from random import randint
from neurose.functions import sigmoid


class TestFunctions(TestCase):

    def test_sigmoid(self):
        rand = [[randint(-10, 10) for i in range(50)] for i in range(30)]
        torch_sigmoid = Sigmoid()
        torch_result = torch_sigmoid(torch.DoubleTensor(rand))
        self.assertTrue((torch_result ==  torch.from_numpy(sigmoid(rand)).double()).all())

