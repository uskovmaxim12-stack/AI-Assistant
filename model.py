# model.py
import numpy as np
from layers import Embedding, Linear, LayerNorm
from transformer import TransformerBlock

class GPT:
    def __init__(self, config):
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = self._create_positional_encoding(config.max_len, config.d_model)
        self.blocks = [TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
                       for _ in range(config.num_layers)]
        self.ln = LayerNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size)

    def _create_positional_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
                if i+1 < d_model:
                    pe[pos, i+1] = np.cos(pos / (10000 ** (2 * i / d_model)))
        return pe

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.shape[1]
        x = self.embedding.forward(x) + self.positional_encoding[:seq_len, :]
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln.forward(x)
        logits = self.lm_head.forward(x)
        return logits
