"""ViT encoder: conv patch-embed -> CLS token + learned pos-enc -> transformer.

Note: with ~187 train windows/person this ViT is data-starved; it is included
for architectural comparison, not as the expected best performer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CONFIG


class ViTEncoder(nn.Module):
    def __init__(
        self,
        in_ch=2,
        img_size=(52, 250),
        patch=(13, 25),
        dim=128,
        depth=4,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        embed_dim=128,
    ):
        super().__init__()
        gh, gw = img_size[0] // patch[0], img_size[1] // patch[1]
        self.n_patches = gh * gw
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.patch_embed(x)              # [B, dim, gh, gw]
        x = x.flatten(2).transpose(1, 2)     # [B, n_patches, dim]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])
        return self.out(cls_out)


def build(cfg=CONFIG):
    v = cfg["vit"]
    shape = cfg["tensor"]["shape"]
    return ViTEncoder(
        in_ch=shape[0],
        img_size=(shape[1], shape[2]),
        patch=tuple(v["patch"]),
        dim=v["dim"],
        depth=v["depth"],
        heads=v["heads"],
        mlp_ratio=v["mlp_ratio"],
        dropout=v["dropout"],
        embed_dim=cfg["embed_dim"],
    )
