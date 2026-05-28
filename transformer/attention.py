import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q,K,V in one matrix (all heads)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # Final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # qkv : (batch_size, seq_len, 3 * embed_dim)
        qkv = self.qkv_proj(x)
        qkv = qkv(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Separate Q, K, and V
        # Each of dimension (batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=3)

        out = attn @ v

        # Recombine the heads
        # out : (batch_size, seq_len, num_heads, dim_heads)
        out = out.transpose(1, 2)

        # out : (batch_size, seq_len, embed_dim)
        out = out.contiguous().view(batch_size, seq_len, embed_dim)

        # Final projection
        out = self.out_proj(out)

        return out
