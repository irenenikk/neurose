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
    no_kernel_locations = int((x.shape[1] - kernel_size)/stride + 1)
    kernel_place = 0
    result_height = int(x.shape[0]*kernel_size**2)
    result_width = int(((x.shape[1] - kernel_size)/stride + 1)**2)
    col_matrix  = np.zeros((result_height, result_width))
    k = kernel_size
    i = 0
    # vertical indexing
    for v in range(no_kernel_locations):
        # horizontal indexing
        for h in range(no_kernel_locations):
            col = x[:, v:k+v, h:k+h].flatten()
            col_matrix[:, i] = col
            i += 1
    return col_matrix, no_kernel_locations
torch.random.manual_seed(666)
inp = np.arange(16).reshape(1, 4, 4)
layer = nn.Conv2d(in_channels=1, out_channels=1,  kernel_size=2, stride=1, bias=False)
output = layer(torch.tensor(inp).unsqueeze(0).float())
kernel = list(layer.parameters())[0][0]
inp_col, _ = im2col(inp, kernel.shape[1], stride=1)
kernel_col, _ = im2col(kernel.detach().numpy(), kernel_size=1, stride=1)
my_output = np.dot(kernel_col, inp_col).reshape(1, 3, 3)
print(output)
print(my_output)
