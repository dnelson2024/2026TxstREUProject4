"""MLP encoder: flatten the [2,52,L] tensor and run 3 dense blocks."""

from __future__ import annotations

import torch.nn as nn

from ..config import CONFIG


class MLPEncoder(nn.Module):
    def __init__(self, in_shape=(2, 52, 250), embed_dim=128, hidden=256, dropout=0.3, layers=3):
        super().__init__()
        in_dim = in_shape[0] * in_shape[1] * in_shape[2]
        self.flatten = nn.Flatten()
        blocks = []
        prev = in_dim
        for _ in range(layers - 1):
            blocks += [
                nn.Linear(prev, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = hidden
        self.body = nn.Sequential(*blocks)
        self.out = nn.Linear(prev, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.flatten(x)
        x = self.body(x)
        return self.out(x)


def build(cfg=CONFIG):
    shape = tuple(cfg["tensor"]["shape"])
    m = cfg["mlp"]
    return MLPEncoder(
        in_shape=shape,
        embed_dim=cfg["embed_dim"],
        hidden=m["hidden"],
        dropout=m["dropout"],
        layers=m["layers"],
    )
