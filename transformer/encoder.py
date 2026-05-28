import attention
import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn_layer = attention.MultiHeadAttention(embed_dim, num_heads)
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.out_layer = nn.Linear(embed_dim, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Sublayer 1
        attn_out = self.attn_layer(x)
        attn_out = self.attn_norm(x + attn_out)

        # Subayer 2
        out = self.out_layer(attn_out)
        out = self.out_norm(out + attn_out)

        return out
