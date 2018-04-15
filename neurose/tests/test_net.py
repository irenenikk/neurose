from unittest import TestCase
import numpy as np
from random import randint
from neurose.layers import Linear
from neurose.net import Net
from neurose.functions import Sigmoid, SoftMax, MeanSquaredError


class TestNet(TestCase):

    def test_calculate_loss_raises_error_if_wrong_dimensions(self):
        pred = np.asarray([[i for i in range (2)] for j in range(3)])
        lab = np.asarray([[i for i in range (3)] for j in range(2)])
        n = Net(MeanSquaredError)
        self.assertRaises(ValueError, n.calculate_loss, pred, lab)

    def test_calculate_loss_sets_final_output_derivative(self):
        pred = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        lab = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        n = Net(MeanSquaredError)
        n.calculate_loss(pred, lab)
        assert not n.loss_derivative == 0

    def test_backpropagation_raises_error_if_calculate_loss_not_called(self):
        pred = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        lab = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        n = Net(MeanSquaredError)
        self.assertRaises(ValueError, n.backpropagate)

    def test_update_weights_raises_error_if_backpropagate_not_called(self):
        pred = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        lab = np.asarray([[randint(0, 3) for i in range (2)] for j in range(3)])
        n = Net(MeanSquaredError)
        self.assertRaises(ValueError, n.update_weights)
