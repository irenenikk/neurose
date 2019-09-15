import numpy as np
from net import Net
from layers.linear import Linear
from functions import Passive, Sigmoid
from functions import MeanSquaredError as MSE
import time
import torch
from random import randint
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F

epochs = 500


class PassivEx(Net):

    def __init__(self):
        super().__init__(MSE, learning_rate=0.02)
        self.a1 = Passive(self)
        self.l1 = Linear(self, 1, 5)
        self.l2 = Linear(self, 5, 1)

    def forward_pass(self, input):
        x = self.a1.call(self.l1.forward(input))
        x = self.a1.call(self.l2.forward(x))
        return x


class SigmoidEx(Net):

    def __init__(self):
        super().__init__(MSE, learning_rate=0.02)
        self.a1 = Sigmoid(self)
        self.l1 = Linear(self, 1, 5)
        self.l2 = Linear(self, 5, 1)

    def forward_pass(self, input):
        x = self.a1.call(self.l1.forward(input))
        x = self.a1.call(self.l2.forward(x))
        return x


class TorchWithActivation(nn.Module):

    def __init__(self):
        super(TorchWithActivation, self).__init__()
        self.l1 = nn.Linear(1, 5)
        self.l2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        return torch.sigmoid(self.l2(x))


def train_with_pytorch(torch_network, input):
    global epochs
    opt = optim.SGD(torch_network.parameters(), lr=0.02)
    time_sum = 0
    x = Variable(torch.FloatTensor(input).unsqueeze(1))
    y = Variable(x.data * 2)
    for epoch in range(epochs):
        start_time = time.time()
        opt.zero_grad()
        y_pred = torch_network(x)
        loss_function = nn.MSELoss()
        torch_loss = loss_function(y_pred, y)
        torch_loss.backward()
        opt.step()
        time_sum += (time.time() - start_time)
    return time_sum / epochs


def train_with_neurose(neurose, input):
    global epochs
    time_sum = 0
    input = np.asarray(input).reshape((len(input), 1))
    label = []
    for i in input:
        for j in i:
            label.append([2 * j])
    actual = np.asarray(label)
    for epoch in range(epochs):
        start_time = time.time()

        neurose.reset_saved_parameters()

        output = neurose.forward(input)

        neurose.calculate_loss(output, actual)

        neurose.backpropagate()

        neurose.update_weights()

        time_sum += (time.time() - start_time)
    return time_sum / epochs

def test_time(batch_size):
    global TorchWithActivation
    input = [randint(0, 10) for i in range(batch_size)]

    e = PassivEx()
    print('without any activation')
    neurose_time = train_with_neurose(e, input)
    # do the same training with torch
    torch_network = nn.Sequential(
        nn.Linear(1, 5),
        nn.Linear(5, 1)
    )
    torch_time = train_with_pytorch(torch_network, input)
    print('    training cycle with neurose took in average {} seconds'.format(neurose_time))
    print('    training cycle with torch took in average {} seconds'.format(torch_time))
    print('    difference: {} seconds'.format(neurose_time - torch_time))
    print('with sigmoid as activation function')
    e_with_activation = SigmoidEx()
    neurose_time_sigmoid = train_with_neurose(e_with_activation, input)
    torch_network_with_activation = TorchWithActivation()
    torch_time_sigmoid = train_with_pytorch(torch_network_with_activation, input)
    print('    training cycle with neurose took in average {} seconds'.format(neurose_time_sigmoid))
    print('    training cycle with torch took in average {} seconds'.format(torch_time_sigmoid))
    print('    difference between neurose and pytorch: {} seconds'.format(neurose_time - torch_time_sigmoid))
    print('    with neurose, using an activation made the process {}% slower'.format(((neurose_time_sigmoid - neurose_time)/neurose_time)*100))
    print('    with torch, using an activation made the process {}% slower'.format(((torch_time_sigmoid - torch_time)/torch_time)*100))


batch = 1
for _ in range(5):
    print('Using batch size of {}'.format(batch))
    print('----------------------------------')
    test_time(batch)
    batch *= 10
    print('----------------------------------')
    print()