"""Encoder registry."""

from ..config import CONFIG
from .....csi_pipeline.models import cnn, mlp, vit
from .heads import LinearProbe, ProjectionHead

_BUILDERS = {"mlp": mlp.build, "cnn": cnn.build, "vit": vit.build}


def build_encoder(name: str, cfg=CONFIG):
    if name not in _BUILDERS:
        raise ValueError(f"unknown encoder '{name}', expected one of {list(_BUILDERS)}")
    return _BUILDERS[name](cfg)


__all__ = ["build_encoder", "ProjectionHead", "LinearProbe"]
