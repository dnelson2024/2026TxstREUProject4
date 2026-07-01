"""CNN encoder: 3 conv blocks over the [2,52,L] image -> global avg pool."""

from __future__ import annotations

import torch.nn as nn

from ..config import CONFIG


class CNNEncoder(nn.Module):
    def __init__(self, in_ch=2, channels=(16, 32, 64), embed_dim=128, dropout=0.2):
        super().__init__()
        blocks = []
        prev = in_ch
        for ch in channels:
            blocks += [
                nn.Conv2d(prev, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            ]
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Linear(prev, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.out(x)


def build(cfg=CONFIG):
    c = cfg["cnn"]
    shape = cfg["tensor"]["shape"]
    return CNNEncoder(
        in_ch=shape[0],
        channels=tuple(c["channels"]),
        embed_dim=cfg["embed_dim"],
        dropout=c["dropout"],
    )
