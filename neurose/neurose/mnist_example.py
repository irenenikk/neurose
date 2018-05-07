import numpy as np
from net import Net
from layers import Linear
from functions import ReLu, CrossEntropy, SoftMax
import torchvision
import torch
import time


class Ex(Net):

    def __init__(self):
        super().__init__(CrossEntropy, learning_rate=0.05)
        self.a1 = ReLu(self)
        self.a2 = SoftMax(self)
        self.l1 = Linear(self, 784, 256)
        self.l2 = Linear(self, 256, 120)
        self.l3 = Linear(self, 120, 64)
        self.l4 = Linear(self, 64, 10)

    def forward_pass(self, input):
        x = input.reshape(input.shape[0], -1)
        x = self.a1.call(self.l1.forward(x))
        x = self.a1.call(self.l2.forward(x))
        x = self.a1.call(self.l3.forward(x))
        x = self.a2.call(self.l4.forward(x))
        return x

def get_prediction(vector):
    preds = np.zero((len(vector), 1))
    max = 0
    max_indx = 0
    for o in vector:
        for i, v in enumerate(vector[o]):
            if v > max:
                max = v
                max_indx = i
        preds[o] = max_indx
    return preds



train_dataset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=256, num_workers=2)

test_dataset = torchvision.datasets.MNIST('./data', download=True, transform=torchvision.transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=2)

e = Ex()

train_start = time.time()
for epoch in range(5):

    epoch_start = time.time()

    for x, y in train_dataloader:

        pass_start = time.time()

        e.reset_saved_parameters()

        input = x.numpy()
        input = input.astype(np.float128)

        actual = y.numpy()

        output = e.forward_pass(input)

        loss = e.calculate_loss(output, actual)

        e.backpropagate()

        e.update_weights()

    print('loss for epoch {}: {}'.format(epoch, loss))
    print('epoch took {} minutes'.format((time.time() - epoch_start)/60))

print('done training')
print('training took {} minutes'.format((time.time() - train_start) / 60))

print('testing')

accuracy = 0

for x, y in test_dataloader:

    e.reset_saved_parameters()

    input = x.numpy()
    input = input.astype(np.float128)
    actual = y.numpy()

    output = e.forward_pass(input)

    preds = get_prediction(output)

    accuracy += sum(actual == preds)

print('Test accuracy: {}'.format(accuracy/len(test_dataloader)))


