import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn_feat = None

    def forward(self, x, *args):
        return x
