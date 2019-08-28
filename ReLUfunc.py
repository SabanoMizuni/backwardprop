"""
forward/backward propagation of the activation function (ReLU layer)
"""

import numpy as np

class ReLUfunc:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # Masking with a boolean matrix
        out = x.copy() # shallow copying
        out[self.mask] = 0

        return out

    def backward(self, d_out):
        d_out[self.mask] = 0
        d_x = d_out

        return d_x