from unittest import TestCase
import numpy as np
import torch.nn as nn
import torch
from neurose.utils import im2col

class TestConvLayer(TestCase):

    def test_im2col(self):
        inp = np.arange(16).reshape(1, 4, 4)
        expected_out = np.asarray([[ 0., 1., 2., 4., 5., 6., 8., 9., 10.],
                        [ 1., 2., 3., 5., 6., 7., 9., 10., 11.],
                        [ 4., 5., 6., 8., 9., 10., 12., 13., 14.],
                        [ 5., 6., 7., 9., 10., 11., 13., 14., 15.]])
        res, _ = im2col(inp, 2, 1)
        print(expected_out)
        self.assertTrue(np.array_equal(expected_out, res))
