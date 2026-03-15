# attention.py
import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, depth)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q = self.Wq.forward(q)
        k = self.Wk.forward(k)
        v = self.Wv.forward(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.depth)
        if mask is not None:
            scores += (mask * -1e9)
        attention_weights = softmax(scores)
        out = np.matmul(attention_weights, v)

        # Concatenate heads
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        out = self.Wo.forward(out)
        return out
