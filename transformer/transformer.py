import torch
import config
from torch import nn


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_enc_layers = nn.ModuleList()
