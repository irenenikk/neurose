import numpy as np
import torch
import torch.nn as nn

def im2col(x, kernel_size, stride):
    """
    :param x Assumed to be a tensor with three dimensions (a, b, c), where b = c. In other words it should be a stack of square matrices.
    :param kernel_size The size of one side of the kernel matrix.
    :param stride The size of the steps the kernel matrix takes after each multiplication.
    :return The input matrix stretched out into one matrix.
    """
    input_size = x.shape[1]
    no_kernel_locations = int((input_size - kernel_size)/stride + 1)
    kernel_place = 0
    result_height = int(x.shape[0]*kernel_size**2)
    result_width = int(((input_size - kernel_size)/stride + 1)**2)
    col_matrix  = np.zeros((result_height, result_width))
    k = kernel_size
    i = 0
    v = 0
    h = 0
    # vertical indexing
    for _ in range(no_kernel_locations):
        # horizontal indexing
        h = 0
        for _ in range(no_kernel_locations):
            col = x[:, v:k+v, h:k+h].flatten()
            col_matrix[:, i] = col
            h += stride
            i += 1
        v += stride
    return col_matrix, no_kernel_locations

def im2row(x):
    """
    :param x Assumed to be a tensor with four dimensions (kernel amount, input channels, kernel size, kernel size).
    :return The input matrix stretched out into one matrixof shape (kernel amount, input channels * kernel size * kernel size).
    """
    kernel_amount, in_channels, kernel_size, kernel_size = x.shape
    output = np.zeros((kernel_amount, kernel_size**2*in_channels))
    for i in range(kernel_amount):
        output[i] = x[i].flatten()
    return output
