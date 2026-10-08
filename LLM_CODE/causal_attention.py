import torch
import torch.nn as nn

class Causal_Attention(nn.Module):

    def __init__(self, In_dim, Out_dim):
        super().__init__()

        self.W_Q = nn.Linear(In_dim, Out_dim)
        self.W_K = nn.Linear(In_dim, Out_dim)
        self.W_V = nn.Linear(In_dim, Out_dim)

    def forward(self, x):

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        attention_score = torch.matmul(Q, K.transpose(-2, -1))


        attention_score = attention_score / (K.shape[-1] ** 0.5)

        mask = torch.triu(
            torch.ones_like(attention_score),
            diagonal=1
        ).bool()
        attention_score = attention_score.masked_fill(
            mask, float('-inf')
        )
        attention_score = torch.softmax(attention_score, dim=-1)
        final_output = torch.matmul(attention_score, V)

      
