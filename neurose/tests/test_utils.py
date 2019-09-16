from unittest import TestCase
import numpy as np
import torch.nn as nn
import torch
from neurose.utils import im2col, im2row

class TestConvLayer(TestCase):

    def test_im2col_with_stride_one(self):
        inp = np.arange(16).reshape(1, 4, 4)
        expected_out = np.asarray([[ 0., 1., 2., 4., 5., 6., 8., 9., 10.],
                                   [ 1., 2., 3., 5., 6., 7., 9., 10., 11.],
                                   [ 4., 5., 6., 8., 9., 10., 12., 13., 14.],
                                   [ 5., 6., 7., 9., 10., 11., 13., 14., 15.]])
        res, _ = im2col(inp, 2, 1)
        self.assertTrue(np.array_equal(expected_out, res))

    def test_im2row_with_kernel_size_one(self):
        inp = np.asarray([[[[-0.1881, -0.2299],
                          [-0.3882, -0.3988]]]])
        expected_out = np.asarray([[-0.1881, -0.2299, -0.3882, -0.3988]])
        res = im2row(inp)
        self.assertTrue(np.array_equal(expected_out, res))

    def test_im2col_with_bigger_stride(self):
        inp = np.arange(16).reshape(1, 4, 4)
        expected_out = np.asarray([[ 0., 2., 8., 10.],
                                   [ 1., 3., 9., 11.],
                                   [ 4., 6., 12., 14.],
                                   [ 5., 7., 13., 15.]])
        res, _ = im2col(inp, 2, 2)
        self.assertTrue(np.array_equal(expected_out, res))

    def test_im2row_with_multiple_kernels(self):
        inp = np.asarray([[[[-0.1881, -0.2299],
                          [-0.3882, -0.3988]]]])
        expected_out = np.asarray([[-0.1881, -0.2299, -0.3882, -0.3988]])
        res = im2row(inp)
        self.assertTrue(np.array_equal(expected_out, res))

