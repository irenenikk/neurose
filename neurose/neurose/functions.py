import numpy as np
from functools import reduce


class DifferentiableFunction:
    """
    Parent class for all activation functions. Is used to store the activation functions of each layer for backpropagation.
    """
    def __init__(self, net):
        self.net = net

    def call(self, x):
        """
        A wrapper for the actual function so we can save stuff for backpropagation.
        """
        self.net.save_activation_function(self)
        output = self.func(x)
        self.net.save_output(output)
        return output

    def func(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

class Passive(DifferentiableFunction):
    """
    When you don't want to use an activation function an a layer, use this. Is also called "linear activation function".
    """
    def func(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)


class Sigmoid(DifferentiableFunction):
    """
    Just your average activation function: https://en.wikipedia.org/wiki/Sigmoid_function
    """

    def func(self, x):
        return 1/(1 + np.exp(-np.array(x)))

    def derivative(self, x):
        s = np.array(self.func(x))
        return s*(1-s)


class SoftMax(DifferentiableFunction):
    """
    Maps values in an array between [0,1]
    Is used mostly in output layer to obtain probabilities
    """

    def func(self, x):
        """
        This had to be hacked a bit, because np.exp overflows quite easily
        Hack from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """
        out = []
        for i in range(len(x)):
            b = np.amax(x[i])
            diff = x[i] - b
            y = np.exp(diff)
            out.append(y/y.sum())
        return np.asarray(out)

    def derivative(self, x):
        s = np.array(self.call(x))
        return s*(1-s)


class ReLu(DifferentiableFunction):
    """
    A surprisingly effective activation function: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """

    def func(self, x):
        return np.maximum(x, 0)

    def derivative(self, x):
        for i, val1 in enumerate(x):
            for j, val2 in enumerate(val1):
                val = x[i][j]
                x[i][j] = 1 if val > 0 else 0
        return x

# Just your average loss function
class MeanSquaredError:

    @staticmethod
    def call(outputs, labels):
        """
        We assume that the input is a matrix with each row representing a different output
        So we calculate the loss for the whole batch.
        :param outputs: Outputs of the neural network on a specific batch. Of shape (batch_size, output_dimension)
        :param labels: The true labels of a specific batch. Of shape (batch_size, output_dimension).
        :return: The mean square error of a specific batch
        """
        if not isinstance(outputs, np.ndarray) or not isinstance(labels, np.ndarray):
            raise ValueError('Loss functions require np arrays as inputs: {}'.format(outputs, labels))
        if not outputs.shape == labels.shape:
            raise ValueError('Outputs and labels are a different shape: {} and {}'.format(outputs.shape, labels.shape))
        if len(outputs) == 0 or len(labels) == 0:
            raise ValueError('No outputs or labels given')
        sum = 0
        for o, l in zip(outputs, labels):
            if not len(o) == len(l): raise ValueError('Outputs and labels are of different dimesion: {} and {}'.format(o, l))
            # Use Euclidean distance
            sum += np.linalg.norm(o - l) ** 2
        # The mean is calculated element wise
        mean = sum/(len(outputs)*len(outputs[0]))
        return mean

    @staticmethod
    def derivative(outputs, labels):
        """
        The derivative of mean square error with regards to the outputs of the neural network.
        Calculations can be found in notes about backpropagation in the wiki. Used in backpropagation.
        """
        batch_size = len(outputs)
        dimension = len(outputs[0])
        return 2/(batch_size*dimension)*(outputs - labels)


class CrossEntropy:

    @staticmethod
    def call(outputs, labels):
        """
        This is implemented in exactly the same way as in Pytorch. See https://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss
        :param outputs: is of shape (b, n), where b is batch size and n is the number of classes.
        :param labels: is of shape (b, )
        """
        if not isinstance(outputs, np.ndarray) or not isinstance(labels, np.ndarray):
            raise ValueError('Loss functions require np arrays as inputs: {}'.format(outputs, labels))
        if not len(outputs) == len(labels):
            raise ValueError('Outputs and labels are a different length: {} and {}'.format(len(outputs), len(labels)))
        if len(outputs) == 0 or len(labels) == 0:
            raise ValueError('No outputs or labels given')
        no_samples = len(outputs)
        sum = 0
        for i in range(len(labels)):
            exp = np.exp(outputs[i])
            log = np.log(np.sum(exp))
            label = labels[i]
            output = outputs[i][label]
            sum += -output + log
        return 1/no_samples * sum

    @staticmethod
    def derivative(outputs, labels):
        """
        See https://deepnotes.io/softmax-crossentropy for calculations
        The labels have to be reshaped so that numpy can broadcast them to match outputs' shape
        """
        return outputs - np.reshape(labels, (len(labels), 1))
