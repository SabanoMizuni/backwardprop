"""
forward/backward propagation of the Matrix Multiplication i.e. W * X
"""

import numpy as np

class Mul_matrix:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.d_W = None
        self.d_b = None

    def forward(self, X):
        self.X = X
        out = np.dot(X, self.W) + self.b

        return out

    def backward(self, d_out):
        d_x = np.dot(d_out, self.W.T)
        self.d_W = np.dot(self.x.T, d_out)
        self.d_b = np.sum(d_out, axis = 0)

        return d_x