import numpy as np
from neurose.utils import im2col

class Conv2D:

    def __init__(self, network, input_channels, kernel_amount, kernel_size, stride=1, padding=0, initial_weights=None, initial_biases=None):
        """
        :param network: The neural network this layer belongs to.
        :param input_channels: Amount of channels in the input image.
        :param kernel_amount: Amount of kernels to use in the layer. Equal to the layer depth.
        :param initial_weights: Initial weights. Random ones are seeded if none given. Giving initial weights makes
        teh functionality of the layer deterministic, which is a requirement for tests.
        :param initial_biases: Initial biases for this layer. Random ones are seeded if none are given.
        :param stride: The size of the stride, or the step that the kernel takes after each convolution.
        :param padding: The size of zero padding added to the sides of the input to include boundary values.
        """
        self.network = network
        if input_channels < 0 or kernel_amount < 0:
            raise ValueError('Input and output sizes have to be positive for the convolution layer')
        self.kernel_size = kernel_size
        self.kernel_amount = kernel_amount
        self.stride = stride
        self.padding = padding
        self.kernel = initial_weights if initial_weights is not None else np.random.random(size=(kernel_amount, input_channels, kernel_size, kernel_size))
        self.biases = np.ndarray(0)

    def forward_pass(self, inp):
        """
        ¨
        :param input: input for a specific layer of shape (batch_size, input_dimension)
        :return: output of this layer after adding weights and biases of shape (batch_size, output_dimension)
        """
        F = self.kernel_size
        W = inp.shape[0] # the input has to be a square matrix
        P = self.padding
        S = self.stride
        if (W - F + 2 * P) % S != 0:
            raise ValueError('Invalid choice of padding, input, kernel and stride size.')
        # use im2col to do convolution as one neat matrix multiplication
        col_vector_size = self.kernel_size**2*self.kernel_amount
        print('self kernel shape', self.kernel.shape)
        inp_col, kernel_locations = im2col(inp, self.kernel.shape[2], stride=1)
        kernel_col, _ = im2col(self.kernel, kernel_size=1, stride=1)
        return np.dot(kernel_col, inp_col).reshape(self.kernel_amount, kernel_locations, kernel_locations)


