"""
forward/backward propagation of the activation function (Sigmoid layer)
"""

import numpy as np

class Sigmoidfunc:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, d_out):
        d_x = d_out * (1.0 - self.out) * self.out

        return d_x