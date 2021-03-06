import numpy as np
from neurose.utils import im2col, im2row

class Conv2D:

    def __init__(self, network, input_channels, kernel_amount, kernel_size, stride=1, padding=0, use_biases=False, initial_weights=None, initial_biases=None):
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
        self.use_biases = use_biases
        if use_biases:
            self.biases = initial_biases if initial_biases is not None else np.zeros(kernel_amount)

    def forward_pass(self, inp):
        """
        ¨
        :param input: input for a specific layer of shape (batch_size, channels, n, n)
        :return: output of this layer after adding weights and biases of shape (batch_size, 1, kernel_locations, kernel_locations), 
                where kernel locations is the amount of times the kernel can be passed accross the input
        """
        if len(inp.shape) < 4:
            raise ValueError('Make sure the input has 4 dimensions: (batch_size, channels, n, n)')
        batch_size = inp.shape[0]
        channels = inp.shape[1]
        F = self.kernel_size
        W = inp.shape[2] # the input has to be a square matrix
        P = self.padding
        S = self.stride
        if (W - F + 2 * P) % S != 0:
            raise ValueError('Invalid choice of padding, input, kernel and stride size.')
        # use im2col to do convolution as one neat matrix multiplication
        col_vector_size = self.kernel_size**2*self.kernel_amount
        if self.padding > 0:
            inp = np.pad(inp, ((0,0), (0,0),(self.padding,self.padding),(self.padding,self.padding)), 'constant')
        # move batches to the bottom in order to just do one matrix multiplication
        inp = np.concatenate(inp, axis=1)
        inp_col, kernel_locations = im2col(inp, self.kernel_size, stride=1)
        kernel_row = im2row(self.kernel)
        if self.use_biases:
            inp_col = np.vstack([inp_col, np.ones(kernel_locations**2*batch_size)])
            kernel_row = np.hstack([kernel_row, self.biases.reshape(self.kernel_amount, 1)])
        # this is a really ugly way of reshaping the dot product result
        # the point is to essentially achieve the output shape
        # but vanilla reshaping wouldn't work properly
        return np.stack(np.dot(kernel_row, inp_col).reshape(self.kernel_amount, batch_size, kernel_locations, kernel_locations), axis=1)


