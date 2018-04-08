from unittest import TestCase
from neurose.layers import Linear
from random import randint
from neurose.net import Net
from neurose.functions import Sigmoid


class TestLinearLayer(TestCase):

    def test_linear_layer_output_correct_shape(self):
        input = randint(5, 10)
        output = randint(5, 10)
        a = [[randint(0, 5)] for i in range(input)]
        linear = Linear(Net(Sigmoid), input, output)
        self.assertTrue(len(linear.forward(a)) == output)