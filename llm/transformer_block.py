from llm.attention import MultiHeadAttention
import torch
import torch.nn as nn


class Transformer_block(nn.Module):

    def __init__(self, d_in, d_out, d_head, context_length):
        super().__init__()

        assert d_in == d_out
        assert d_out % d_head == 0

        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.d_head = d_head

        self.num_heads = d_out // d_head

        # Multi-Head Self Attention
        self.multihead_attention = MultiHeadAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=0.1,
            num_heads=self.num_heads,
            qkv_bias=True
        )

        # Two separate LayerNorms
        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_out)

        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_out, 4 * d_out),
            nn.GELU(),
            nn.Linear(4 * d_out, d_out)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        # First residual connection
        shortcut = x

        normalised_input = self.norm1(x)

        attention = self.multihead_attention(normalised_input)

        attention = self.dropout(attention)

        x = attention + shortcut

        # Second residual connection
        shortcut = x

        normalized_2 = self.norm2(x)

        forwardx = self.feed_forward(normalized_2)

        forwardx = self.dropout(forwardx)

        x = forwardx + shortcut

        return x


if __name__ == "__main__":

    model = Transformer_block(
        d_in=768,
        d_out=768,
        d_head=64,
        context_length=1024
    )

    x = torch.randn(2, 16, 768)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
