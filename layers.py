# layers.py
import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - self.mean) / np.sqrt(self.var + self.eps) + self.beta

    def backward(self, dout):
        # Упрощённо, для демонстрации
        return dout

class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, dout):
        return dout * self.mask

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.W = np.random.randn(vocab_size, d_model) * 0.01
        self.dW = np.zeros_like(self.W)

    def forward(self, x):
        self.x = x
        return self.W[x]

    def backward(self, dout):
        np.add.at(self.dW, self.x, dout)
        return None
