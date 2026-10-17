import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )

    def forward(self, x):

        # Input shape: (batch_size, num_tokens, d_in)
        b, num_tokens, d_in = x.shape

        # Generate Q, K, V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split into multiple attention heads
        # (b, num_tokens, d_out)
        # -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim)
        # -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculate attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scaled dot-product attention
        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5),
            dim=-1
        )

        attn_weights = self.dropout(attn_weights)

        # Multiply attention weights with values
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Concatenate attention heads
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        # Output projection
        context_vec = self.out_proj(context_vec)

        return context_vec


if __name__ == "__main__":

    torch.manual_seed(42)

    # GPT-2 Small configuration
    d_in = 768
    d_out = 768
    context_length = 1024
    num_heads = 12
    dropout = 0.1

    # Initialize model
    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        qkv_bias=True
    )

    # Dummy input
    # Batch size = 2
    # Sequence length = 100
    # Embedding dimension = 768
    x = torch.randn(2, 100, 768)

    # Set evaluation mode (disables dropout)
    mha.eval()

    # Forward pass
    with torch.no_grad():
        output = mha(x)

    # Print results
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print("\nNumber of attention heads:", mha.num_heads)
    print("Head dimension:", mha.head_dim)

    # Check output dimensions
    assert output.shape == (2, 100, 768)

    print("\nMulti-Head Attention test passed!")