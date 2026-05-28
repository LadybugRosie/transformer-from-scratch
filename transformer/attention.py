import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0, is_masked=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_masked = is_masked

        # Q,K,V in one matrix (all heads)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # Final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # qkv : (batch_size, seq_len, 3 * embed_dim)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Separate Q, K, and V
        # Each of dimension (batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if self.is_masked:
            mask = torch.tril(torch.ones_like(scores))
            scores = torch.masked_fill(scores, mask, float("-inf"))

        attn = torch.softmax(scores, dim=3)

        scores = self.dropout(scores)

        out = attn @ v

        # Recombine the heads
        # out : (batch_size, seq_len, num_heads, dim_heads)
        out = out.transpose(1, 2)

        # out : (batch_size, seq_len, embed_dim)
        out = out.contiguous().view(batch_size, seq_len, embed_dim)

        # Final projection
        out = self.out_proj(out)

        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q in one matrix (all heads)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # K, V in one matrix (all heads)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)

        # Final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, dec, enc):
        batch_size, dec_seq_len, embed_dim = dec.shape
        _, enc_seq_len, _ = enc.shape

        # q : (batch_size, dec_seq_len, embed_dim)
        q = self.q_proj(dec)
        q = q.view(batch_size, dec_seq_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        # kv : (batch_size, enc_seq_len, 2 * embed_dim)
        kv = self.kv_proj(enc)
        kv = kv.view(batch_size, enc_seq_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)

        # Separate K, and V
        # Each of dimension (batch_size, num_heads, enc_seq_len, head_dim)
        k, v = kv[0], kv[1]

        # scores : (batch_size, num_heads, dec_seq_len, enc_seq_len)
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        attn = torch.softmax(scores, dim=3)

        scores = self.dropout(scores)

        # out : (batch_size, num_heads, dec_seq_len, dim_heads)
        out = attn @ v

        # Recombine the heads
        # out : (batch_size, dec_seq_len, num_heads, dim_heads)
        out = out.transpose(1, 2)

        # out : (batch_size, dec_seq_len, embed_dim)
        out = out.contiguous().view(batch_size, dec_seq_len, embed_dim)

        # Final projection
        out = self.out_proj(out)

        return out
