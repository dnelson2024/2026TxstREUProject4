"""Shared projection head (for SimCLR) and linear probe (for evaluation)."""

from __future__ import annotations

import torch.nn as nn


class ProjectionHead(nn.Module):
    """2-layer MLP projection head used during contrastive pretraining."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LinearProbe(nn.Module):
    """Single linear layer mapping frozen embeddings to class logits."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)
