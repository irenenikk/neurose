from unittest import TestCase
import numpy as np
from random import randint
from neurose.layers.linear import Linear
from neurose.net import Net
from neurose.functions import Sigmoid


class TestLinearLayer(TestCase):

    def test_linear_layer_output_correct_shape(self):
        input = randint(5, 10)
        output = randint(5, 10)
        a = [randint(0, 5) for i in range(input)]
        linear = Linear(Net(Sigmoid), input, output)
        self.assertTrue(len(linear.forward(a)) == output)

    def test_weights_are_initialized_correctly(self):
        input = randint(5, 10)
        output = randint(5, 10)
        weights = np.asarray([[randint(0, 2) for j in range(input)] for i in range(output)])
        linear = Linear(Net(Sigmoid), input, output, weights)
        self.assertTrue((linear.weights == weights).all())

    def test_biases_are_initialized_correctly(self):
        input = randint(5, 10)
        output = randint(5, 10)
        biases = np.asarray([randint(1, 5) for i in range(output)])
        linear = Linear(Net(Sigmoid), input, output, np.ndarray(0), biases)
        self.assertTrue(np.array_equal(linear.biases, biases))

    def test_raises_error_if_weights_not_right_dimension(self):
        input = randint(5, 10)
        output = randint(5, 10)
        weights = np.ndarray((output + 1, input))
        self.assertRaises(ValueError, Linear, Net(Sigmoid), input, output, weights)

    def test_raises_error_if_biases_not_right_dimension(self):
        input = randint(5, 10)
        output = randint(5, 10)
        # biases should be the dimension of output
        biases = np.ndarray(output + 2)
        self.assertRaises(ValueError, Linear, Net(Sigmoid), input, output, np.ndarray(0), biases)

    def test_saves_inputs_outputs_and_weights_to_network(self):
        input = randint(5, 10)
        output = randint(5, 10)
        weights = np.asarray([[randint(0, 2) for j in range(input)] for i in range(output)])
        n = Net(Sigmoid)
        linear = Linear(n, input, output, weights)
        linear.forward(input)
        assert len(linear.network.saved_inputs) > 0
        assert len(linear.network.saved_outputs) > 0
        assert len(linear.network.saved_weights) > 0

    def test_old_weights_are_updated_on_second_pass(self):
        input = randint(5, 10)
        output = randint(5, 10)
        weights = np.asarray([[randint(0, 2) for j in range(input)] for i in range(output)])
        n = Net(Sigmoid)
        linear = Linear(n, input, output, weights)
        linear.forward(input)
        sh = np.asarray(linear.network.saved_weights).shape
        linear.forward(input)
        assert sh == np.asarray(linear.network.saved_weights).shape



