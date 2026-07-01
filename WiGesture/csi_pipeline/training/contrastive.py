"""NT-Xent (SimCLR) loss and CSI-appropriate augmentations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent loss for a batch of 2B embeddings.

    Convention: z = concat([view1 (B rows), view2 (B rows)], dim=0). The positive
    of row i (i < B) is row i+B and vice versa.
    """
    two_b = z.size(0)
    b = two_b // 2
    z = F.normalize(z, dim=1)
    sim = (z @ z.t()) / temperature  # [2B, 2B]

    # Mask self-similarity (use a large negative, AMP-safe).
    diag = torch.eye(two_b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag, -1e9)

    targets = torch.arange(two_b, device=z.device)
    targets = (targets + b) % two_b  # i -> i+B (mod 2B)
    return F.cross_entropy(sim, targets)


def _rand(shape, device):
    return torch.rand(shape, device=device)


def augment(x: torch.Tensor, aug: dict) -> torch.Tensor:
    """Return one stochastic view of a batch x [B,2,52,L].

    Channel 0 = amplitude, channel 1 = phase. Phase is augmented more gently
    when aug['phase_gentler'] is set.
    """
    b, c, n_sc, length = x.shape
    device = x.device
    out = x.clone()

    # Per-channel Gaussian noise.
    noise_std = aug["noise_std"]
    noise = torch.randn_like(out) * noise_std
    if aug.get("phase_gentler", True) and c >= 2:
        noise[:, 1] *= 0.5
    out = out + noise

    # Amplitude scaling (channel 0 only).
    lo, hi = aug["amp_scale"]
    scale = lo + (hi - lo) * _rand((b, 1, 1, 1), device)
    out[:, 0:1] = out[:, 0:1] * scale

    # Random circular time shift.
    max_shift = int(aug["time_shift"])
    if max_shift > 0:
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        out = torch.roll(out, shifts=shift, dims=3)

    # Contiguous time masking.
    tmf = aug["time_mask_frac"]
    if tmf > 0:
        mlen = max(1, int(length * tmf))
        start = int(torch.randint(0, max(1, length - mlen + 1), (1,)).item())
        out[:, :, :, start : start + mlen] = 0.0

    # Random subcarrier masking.
    smf = aug["subcarrier_mask_frac"]
    if smf > 0:
        n_mask = max(1, int(n_sc * smf))
        idx = torch.randperm(n_sc, device=device)[:n_mask]
        out[:, :, idx, :] = 0.0

    return out


def two_views(x: torch.Tensor, aug: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return augment(x, aug), augment(x, aug)
