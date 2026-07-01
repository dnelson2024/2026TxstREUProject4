"""Sliding windows, leakage-safe temporal split, and tensor construction.

Each recording is cleaned (full series) before windowing. Windows are split
TEMPORALLY per recording (first 80% train, next 10% val, last 10% test) with a
guard gap so no raw-time content is shared across splits. Z-score statistics are
fit on the train split only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CONFIG
from . import preprocess as pp
from .loader import Recording


@dataclass
class Window:
    amp: np.ndarray    # [L, 52]
    phase: np.ndarray  # [L, 52]
    label: str
    src_recording_id: str
    window_index: int
    start: int  # raw-sample start index within the recording


def sliding_windows(rec: Recording, win: int, stride: int) -> list[Window]:
    """Slice one cleaned recording into overlapping windows."""
    n = rec.amp.shape[0]
    windows: list[Window] = []
    wi = 0
    for start in range(0, n - win + 1, stride):
        windows.append(
            Window(
                amp=rec.amp[start : start + win],
                phase=rec.phase[start : start + win],
                label=rec.gesture_label,
                src_recording_id=rec.src_recording_id,
                window_index=wi,
                start=start,
            )
        )
        wi += 1
    return windows


def temporal_split(
    windows: list[Window],
    train: float,
    val: float,
    test: float,
    guard: int,
) -> tuple[list[Window], list[Window], list[Window]]:
    """Split windows of a SINGLE recording into contiguous train/val/test blocks.

    Drops `guard` windows on each side of a block boundary so overlapping windows
    never straddle two splits (prevents raw-time leakage).
    """
    n = len(windows)
    if n == 0:
        return [], [], []
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    # test gets the remainder
    tr = windows[:n_train]
    va = windows[n_train : n_train + n_val]
    te = windows[n_train + n_val :]

    if guard > 0:
        tr = tr[: max(0, len(tr) - guard)] if va or te else tr
        va = va[guard:] if va else va
        va = va[: max(0, len(va) - guard)] if te else va
        te = te[guard:] if te else te
    return tr, va, te


def split_person(recordings: list[Recording], cfg: dict = None):
    """Window every recording, split each temporally, then pool across gestures.

    Returns three lists of Window (train/val/test), each containing all 6 classes.
    """
    cfg = cfg or CONFIG
    win = cfg["window"]["length"]
    stride = cfg["window"]["stride"]
    guard = cfg["window"]["guard_windows"]
    s = cfg["split"]

    train, val, test = [], [], []
    for rec in recordings:
        ws = sliding_windows(rec, win, stride)
        tr, va, te = temporal_split(ws, s["train"], s["val"], s["test"], guard)
        train += tr
        val += va
        test += te
    return train, val, test


def fit_zscore_stats(train: list[Window]):
    """Fit per-subcarrier z-score stats for amp and phase from train windows."""
    amp_rows = np.concatenate([w.amp for w in train], axis=0)
    phase_rows = np.concatenate([w.phase for w in train], axis=0)
    amp_mean, amp_std = pp.fit_zscore(amp_rows)
    phase_mean, phase_std = pp.fit_zscore(phase_rows)
    return {"amp": (amp_mean, amp_std), "phase": (phase_mean, phase_std)}


def window_to_tensor(w: Window, stats: dict | None) -> np.ndarray:
    """Build a [2, 52, L] tensor: channel 0 = amplitude, 1 = phase.

    Applies z-score with the provided (train-fit) stats when given.
    """
    amp, phase = w.amp, w.phase
    if stats is not None:
        amp = pp.apply_zscore(amp, *stats["amp"])
        phase = pp.apply_zscore(phase, *stats["phase"])
    # [L, 52] -> [52, L]
    return np.stack([amp.T, phase.T], axis=0).astype(np.float32)


def windows_to_arrays(windows: list[Window], stats: dict | None, classes: list[str]):
    """Vectorize a window list to (X[N,2,52,L], y[N]) with integer labels."""
    label_to_idx = {c: i for i, c in enumerate(classes)}
    X = np.stack([window_to_tensor(w, stats) for w in windows], axis=0)
    y = np.array([label_to_idx[w.label] for w in windows], dtype=np.int64)
    return X, y
