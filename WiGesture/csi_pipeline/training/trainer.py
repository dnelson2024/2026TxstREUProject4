"""SimCLR pretraining, encoder freezing, and supervised linear-probe training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import CONFIG
from ..models import LinearProbe, ProjectionHead
from . import contrastive as cl


def make_loader(X: np.ndarray, y: np.ndarray | None, batch_size: int, shuffle: bool, drop_last=False):
    xt = torch.from_numpy(X)
    if y is None:
        ds = TensorDataset(xt)
    else:
        ds = TensorDataset(xt, torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def pretrain_contrastive(encoder, X_train, cfg=CONFIG, device="cpu", val_X=None, verbose=True):
    """SimCLR contrastive pretraining on unlabeled train windows.

    Returns (encoder, history) where history has 'train_loss' and 'val_loss'.
    """
    sc = cfg["simclr"]
    encoder = encoder.to(device)
    proj = ProjectionHead(encoder.embed_dim, cfg["proj_hidden"], cfg["proj_dim"]).to(device)
    params = list(encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=sc["lr"], weight_decay=sc["weight_decay"])

    loader = make_loader(X_train, None, sc["batch_size"], shuffle=True, drop_last=sc["drop_last"])
    val_loader = (
        make_loader(val_X, None, sc["batch_size"], shuffle=False, drop_last=True)
        if val_X is not None and len(val_X) >= 2
        else None
    )

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(sc["epochs"]):
        encoder.train()
        proj.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            v1, v2 = cl.two_views(xb, sc["aug"])
            z = proj(encoder(torch.cat([v1, v2], dim=0)))
            loss = cl.nt_xent_loss(z, sc["temperature"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))

        if val_loader is not None:
            encoder.eval()
            proj.eval()
            vlosses = []
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    v1, v2 = cl.two_views(xb, sc["aug"])
                    z = proj(encoder(torch.cat([v1, v2], dim=0)))
                    vlosses.append(cl.nt_xent_loss(z, sc["temperature"]).item())
            history["val_loss"].append(float(np.mean(vlosses)) if vlosses else float("nan"))

        if verbose and (epoch % 20 == 0 or epoch == sc["epochs"] - 1):
            v = history["val_loss"][-1] if history["val_loss"] else float("nan")
            print(f"    [pretrain] epoch {epoch:3d}  train {history['train_loss'][-1]:.4f}  val {v:.4f}")
    return encoder, history


def freeze(encoder):
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_embeddings(encoder, X, device="cpu", batch_size=256):
    encoder.eval()
    loader = make_loader(X, None, batch_size, shuffle=False)
    out = []
    for (xb,) in loader:
        out.append(encoder(xb.to(device)).cpu())
    return torch.cat(out, dim=0).numpy()


def train_linear_probe(encoder, X_train, y_train, X_val, y_val, n_classes, cfg=CONFIG, device="cpu", verbose=True):
    """Train a linear probe on FROZEN encoder embeddings (supervised CE)."""
    lp = cfg["linear_probe"]
    probe = LinearProbe(encoder.embed_dim, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lp["lr"], weight_decay=lp["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    # Precompute frozen embeddings once.
    Z_train = torch.from_numpy(extract_embeddings(encoder, X_train, device))
    yt = torch.from_numpy(y_train)
    train_loader = DataLoader(TensorDataset(Z_train, yt), batch_size=lp["batch_size"], shuffle=True)

    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        Z_val = torch.from_numpy(extract_embeddings(encoder, X_val, device)).to(device)
        yv = torch.from_numpy(y_val).to(device)

    best_state, best_acc = None, -1.0
    for epoch in range(lp["epochs"]):
        probe.train()
        for zb, yb in train_loader:
            zb, yb = zb.to(device), yb.to(device)
            loss = loss_fn(probe(zb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if has_val:
            probe.eval()
            with torch.no_grad():
                acc = (probe(Z_val).argmax(1) == yv).float().mean().item()
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    if verbose and has_val:
        print(f"    [probe] best val acc {best_acc:.3f}")
    return probe
