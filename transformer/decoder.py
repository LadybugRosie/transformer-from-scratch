import attention
from torch import nn


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        super().__init__()

        self.decode_sublayer = nn.ModuleDict(
            {
                "attention": attention.MultiHeadSelfAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    is_masked=True,
                ),
                "norm": nn.LayerNorm(embed_dim),
            }
        )

        self.encode_sublayer = nn.ModuleDict(
            {
                "attention": attention.MultiHeadCrossAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                ),
                "norm": nn.LayerNorm(embed_dim),
            }
        )

        self.out_sublayer = nn.ModuleDict(
            {
                "mlp": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                ),
                "norm": nn.LayerNorm(embed_dim),
            }
        )

    def forward(self, enc_tkns, dec_tkns):
        # Masked attention layer
        decode_out = self.decode_sublayer["attention"](dec_tkns)
        decode_out = self.decode_sublayer["norm"](dec_tkns + decode_out)

        # Multi-Head attention layer
        cross_out = self.decode_sublayer["attention"](dec_tkns, enc_tkns)
        cross_out = self.decode_sublayer["norm"](cross_out + decode_out)

        # Out layer
        out = self.out_sublayer["mlp"](cross_out)
        out = self.out_sublayer["norm"](cross_out + out)

        return out
