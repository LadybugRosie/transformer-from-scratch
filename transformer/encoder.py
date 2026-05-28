import attention
import torch
from torch import nn

from config import DROPOUT_RATE


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=DROPOUT_RATE):
        # Attention layer
        self.attn_layer = attention.MultiHeadSelfAttention(
            embed_dim, num_heads, dropout_rate=DROPOUT_RATE, is_masked=False
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Feedforward MLP
        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Sublayer 1
        attn_out = self.attn_layer(x)
        attn_out = self.attn_norm(x + attn_out)

        # Sublayer 2
        out = self.out_mlp(attn_out)
        out = self.out_norm(out + attn_out)

        return out
