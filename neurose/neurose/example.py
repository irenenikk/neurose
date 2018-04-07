from neurose.net import Net
from neurose.layers import Linear
from neurose.functions import Sigmoid as s
from neurose.functions import SoftMax as p


class Ex(Net):

    def __init__(self):
        super().__init__()
        # This is where the layers are defined
        self.l1 = Linear(4,3)
        self.l2 = Linear(3,2)

    def forward_pass(self, input):
        # this is where forward pass is defined
        # you define which activation functions to use on which layer
        # and in which order will the layers be traversed
        x = s.call(self.l1.forward(input))
        x = p.call(self.l2.forward(x))
        return x


e = Ex()

# the input consists of three batches, and inputs of dimension 4
output = e.forward([[1, 2, 3, 4], [5, 6, 7, 8], [4, 5, 6, 78]])

print(output)