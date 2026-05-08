import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Minimal placeholder network
        # (does nothing meaningful but is valid)
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x)