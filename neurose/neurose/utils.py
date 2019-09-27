import numpy as np
import torch
import torch.nn as nn

def im2col(x, kernel_size, stride):
    """
    :param x: Assumed to be a tensor with three dimensions (a, b, c). In other words it should be a stack of matrices.
    :param kernel_size: The size of one side of the kernel matrix.
    :param stride: The size of the steps the kernel matrix takes after each multiplication.
    :return The input matrix stretched out into one matrix.
    """
    input_rows = x.shape[1]
    input_cols = x.shape[2]
    batch_size = int(input_rows/input_cols)
    vertical_kernel_locations = int((input_rows - kernel_size)/stride + 1)
    horizontal_kernel_locations = int((input_cols - kernel_size)/stride + 1)
    kernel_locations = int((input_cols - kernel_size)/stride + 1)
    kernel_place = 0
    result_height = int(x.shape[0]*kernel_size**2)
    result_width = kernel_locations**2*batch_size
    col_matrix  = np.zeros((result_height, result_width))
    k = kernel_size
    col_i = 0
    v = 0
    h = 0
    input_i = 1
    # vertical indexing
    for _ in range(batch_size*kernel_locations):
        # horizontal indexing
        h = 0
        for _ in range(kernel_locations):
            col = x[:, v:k+v, h:k+h].flatten()
            col_matrix[:, col_i] = col
            h += stride
            col_i += 1
        # jump to the next input matrix
        if (v + k) % input_cols == 0:
            v = input_i*input_cols
            input_i += 1
        else:
            v += stride
    return col_matrix, horizontal_kernel_locations

def im2row(x):
    """
    In practice, this method is used to stretch out the kernels used in convolution.
    :param x: Assumed to be a tensor with four dimensions (kernel amount, input channels, kernel size, kernel size).
    :return The input matrix stretched out into one matrixof shape (kernel amount, input channels * kernel size * kernel size).
    """
    kernel_amount, in_channels, kernel_size, kernel_size = x.shape
    output = np.zeros((kernel_amount, kernel_size**2*in_channels))
    for i in range(kernel_amount):
        output[i] = x[i].flatten()
    return output
