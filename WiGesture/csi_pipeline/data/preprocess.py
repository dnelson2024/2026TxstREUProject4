"""Full-recording signal cleaning. No windowing happens here.

All functions operate on [N, 52] arrays (time x subcarrier). Order:
  amplitude: hampel -> butterworth low-pass -> (later) z-score
  phase:     linear-fit removal across subcarriers -> unwrap -> savgol smooth
Z-score must be fit on TRAIN windows only, so it is fit/transform separated and
applied at the windowing/split stage, not here.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from ..config import CONFIG


def hampel_filter(x: np.ndarray, window: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Per-subcarrier rolling-median outlier replacement (MAD based)."""
    x = np.asarray(x, dtype=np.float64).copy()
    n, k = x.shape
    half = window // 2
    pad = np.pad(x, ((half, half), (0, 0)), mode="edge")
    # Sliding windows: [n, window, k]
    idx = np.arange(n)[:, None] + np.arange(window)[None, :]
    win = pad[idx]  # [n, window, k]
    med = np.median(win, axis=1)  # [n, k]
    mad = np.median(np.abs(win - med[:, None, :]), axis=1)  # [n, k]
    sigma = 1.4826 * mad
    diff = np.abs(x - med)
    mask = (sigma > 0) & (diff > n_sigmas * sigma)
    x[mask] = med[mask]
    return x


def butter_lowpass(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass, applied per subcarrier (column)."""
    nyq = fs / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, wn, btype="low")
    # filtfilt needs length > 3*(max(len(a),len(b))-1); guard short signals.
    padlen = 3 * (max(len(a), len(b)) - 1)
    if x.shape[0] <= padlen:
        return x
    return filtfilt(b, a, x, axis=0)


def phase_sanitize(phase: np.ndarray, cfg: dict = None) -> np.ndarray:
    """Clean phase: remove per-timestep linear slope across subcarriers,
    unwrap along time, then Savitzky-Golay smooth."""
    cfg = cfg or CONFIG["preprocess"]["phase"]
    phase = np.asarray(phase, dtype=np.float64).copy()
    n, k = phase.shape

    if cfg.get("linear_fit_removal", True):
        # Fit phase vs subcarrier index per timestep; subtract the fitted line.
        sc = np.arange(k)
        # Vectorized least-squares slope/intercept per row.
        sc_mean = sc.mean()
        sc_c = sc - sc_mean
        denom = np.sum(sc_c ** 2)
        p_mean = phase.mean(axis=1, keepdims=True)
        slope = (phase - p_mean) @ sc_c / denom  # [n]
        fit = slope[:, None] * sc_c[None, :] + p_mean
        phase = phase - fit

    phase = np.unwrap(phase, axis=0)

    w = cfg.get("savgol_window", 11)
    p = cfg.get("savgol_poly", 3)
    if w % 2 == 0:
        w += 1
    if w <= n and p < w:
        phase = savgol_filter(phase, w, p, axis=0)
    return phase


def amplitude_pipeline(amp: np.ndarray, fs: float, cfg: dict = None) -> np.ndarray:
    """Hampel -> Butterworth low-pass on raw amplitude (no z-score here)."""
    cfg = cfg or CONFIG["preprocess"]
    amp = hampel_filter(amp, **cfg["hampel"])
    amp = butter_lowpass(amp, fs=fs, **cfg["butter"])
    return amp


def phase_pipeline(phase: np.ndarray, cfg: dict = None) -> np.ndarray:
    return phase_sanitize(phase, (cfg or CONFIG["preprocess"])["phase"])


def fit_zscore(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-subcarrier mean/std fit on `x` ([M,52] stacked train rows)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std
