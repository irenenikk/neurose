from unittest import TestCase
from neurose.layers.linear import Linear
from random import randint


class TestMatrixMath(TestCase):

    def test_linear_layer_output_correct_shape(self):
        input = randint(5, 10)
        output = randint(5, 10)
        a = [[randint(0, 5)] for i in range(input)]
        linear = Linear(input, output)
        self.assertTrue(len(linear.forward(a)) == output)
