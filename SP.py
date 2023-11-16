import numpy as np
from model import *
from optimizer import GradientDescent

nn = NeuralNetwork(3, 2, [4])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# add column of ones to X
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
print(nn.predict(X, [sigmoid, sigmoid]))
y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
print(nn.loss(X, y, [sigmoid, sigmoid]))
opt = GradientDescent(nn, alpha=.3)
activation_functions = [sigmoid, sigmoid]


opt.optimize_full_batch(X, y, [sigmoid, sigmoid])