# transformer.py
import numpy as np
from layers import LayerNorm, Dropout, Linear
from attention import MultiHeadAttention

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu = lambda x: np.maximum(0, x)

    def forward(self, x):
        return self.linear2.forward(self.relu(self.linear1.forward(x)))

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + self.dropout.forward(attn_out))
        ff_out = self.ff.forward(x)
        x = self.norm2.forward(x + self.dropout.forward(ff_out))
        return x
