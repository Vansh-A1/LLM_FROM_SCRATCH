from llm.transformer_block import Transformer_block
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, d_in, d_out, d_head, context_length, vocab_size):
        super().__init__()

        assert d_in == d_out

        self.transformer = nn.ModuleList([
            Transformer_block(d_in, d_out, d_head, context_length)
            for _ in range(12)
        ])

        self.norm = nn.LayerNorm(d_out)

        self.linear = nn.Linear(d_out, vocab_size, bias=False)

    def forward(self, x):

        for block in self.transformer:
            x = block(x)

        x = self.norm(x)

        x = self.linear(x)

        return x
