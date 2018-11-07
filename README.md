# Neurose
#### A simple neural network library for your simple neural network needs.

All the weekly reports and documentation is in the [wiki](https://github.com/irenenikk/neurose/wiki). There you will also find some calculations and notes about neural networks and the algorithms used.

## Current features:
- Linear layer with biases
- Activation functions:
  - Sigmoid
  - ReLu
  - SoftMax
  - Passive (no activation)
- Loss functions
  - Mean squared error
  - Cross Entropy Loss 

## Examples

There are two example models: One learning a linear regression function and another, more complete one classifying the MNIST dataset

Linear regression:

The example model is just overfitting to a simple linear regression problem to prove that the model can learn something. The input is currently `[1, 2, 3, 4]` and the true labels `[2, 4, 6, 8]`. Feel free to toy around with the amount of trianing epochs (iterations of the training loop).

If the weights are initialized with `np.random.normal`, the model sometimes wanders off to a completely wrong direction, which results in some infs and nans. I really don't know why. At the moment the weights are initialized with `np.random.random`, which doesn't result in this problem.

MNIST dataset:

The example uses Pytorch's [MNIST dataset](https://pytorch.org/docs/master/torchvision/datasets.html#mnist), which is downloaded to the subfolder `data` once you run the model.

## Running the example models

0. Make sure you have python 3.x

1. After cloning the project, install depedencies with `pip install -r path/to/requirements.txt`

2. Run the example code with `python path/to/example.py`

3. The program will print the loss for each epoch.

### If you find any errors or problems in this project, all comments and contributions are appreciated!
