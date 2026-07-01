"""Load and parse ESP32 CSI CSV recordings into amplitude / phase arrays.

The dataset CSVs have 28 columns. The CSI itself lives in the `data` column as a
JSON-style list of 104 ints = 52 subcarriers x interleaved I/Q (I0,Q0,I1,Q1,...).
The gesture label is the constant string in the `taget` column (misspelled in the
source data). CSIKit is NOT used -- the data is already raw CSI in the CSV.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CONFIG

_N_IQ = 104  # 52 subcarriers x 2 (I, Q)


@dataclass
class Recording:
    person: str
    gesture_label: str
    amp: np.ndarray   # [N, 52]
    phase: np.ndarray  # [N, 52]
    fs: float
    src_recording_id: str  # e.g. "ID1/applause"


def _parse_data_cell(cell: str) -> list[int] | None:
    """Parse one `data` cell into a list of ints, or None if malformed."""
    try:
        vals = json.loads(cell)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(vals, list) or len(vals) != _N_IQ:
        return None
    return vals


def parse_csi_csv(path: str, label_column: str = None) -> tuple[np.ndarray, str, np.ndarray]:
    """Read one CSV.

    Returns:
        iq:    [N, 104] float array of interleaved I/Q
        label: gesture label string (from the `taget` column)
        ts:    [N] array of pandas Timestamps (parsed from `timestamp`)
    Malformed rows (bad `data` length / parse failure) are dropped and logged.
    """
    label_column = label_column or CONFIG["label_column"]
    df = pd.read_csv(path)

    parsed = df["data"].map(_parse_data_cell)
    good = parsed.notna()
    n_bad = int((~good).sum())
    if n_bad:
        print(f"  [loader] {os.path.basename(path)}: dropped {n_bad} malformed rows")
    df = df[good].reset_index(drop=True)
    iq = np.array(parsed[good].tolist(), dtype=np.float64)  # [N, 104]

    labels = df[label_column].astype(str)
    label = labels.mode().iat[0]  # one gesture per file; mode is robust

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    return iq, label, ts.to_numpy()


def iq_to_amp_phase(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """De-interleave [N,104] I/Q into amplitude and phase, each [N,52]."""
    I = iq[:, 0::2]
    Q = iq[:, 1::2]
    amp = np.hypot(I, Q)
    phase = np.arctan2(Q, I)
    return amp, phase


def resample_uniform(
    amp: np.ndarray,
    phase: np.ndarray,
    timestamps: np.ndarray,
    target_fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate amp/phase onto a uniform `target_fs` time grid.

    Phase is unwrapped along time BEFORE interpolation (interpolating across a
    +/-pi wrap is invalid), then re-wrapped after.
    """
    # Unit-agnostic conversion to seconds (timestamps may be us/ns datetime64).
    ts64 = pd.to_datetime(timestamps).to_numpy().astype("datetime64[ns]")
    t = (ts64 - ts64[0]) / np.timedelta64(1, "s")  # seconds since start
    # Guard against non-monotonic timestamps from clock jitter.
    order = np.argsort(t, kind="stable")
    t = t[order]
    amp = amp[order]
    phase = phase[order]
    # Drop duplicate timestamps (np.interp requires strictly increasing xp).
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, amp, phase = t[keep], amp[keep], phase[keep]

    t0, t1 = t[0], t[-1]
    n_new = max(2, int(round((t1 - t0) * target_fs)) + 1)
    t_new = np.linspace(t0, t1, n_new)

    amp_u = np.empty((n_new, amp.shape[1]))
    phase_u = np.empty((n_new, phase.shape[1]))
    phase_unwrapped = np.unwrap(phase, axis=0)
    for k in range(amp.shape[1]):
        amp_u[:, k] = np.interp(t_new, t, amp[:, k])
        pu = np.interp(t_new, t, phase_unwrapped[:, k])
        phase_u[:, k] = np.angle(np.exp(1j * pu))  # re-wrap to [-pi, pi]
    return amp_u, phase_u


def load_person_recordings(person_id: str, data_root: str = None) -> list[Recording]:
    """Load and parse all dynamic-gesture recordings for one person."""
    data_root = data_root or CONFIG["data_root"]
    target_fs = CONFIG["resample"]["target_fs"]
    do_resample = CONFIG["resample"]["enabled"]
    person_dir = os.path.join(data_root, person_id)

    recordings: list[Recording] = []
    for gesture in CONFIG["classes"]:
        path = os.path.join(person_dir, f"{gesture}.csv")
        if not os.path.exists(path):
            print(f"  [loader] missing {path}, skipping")
            continue
        iq, file_label, ts = parse_csi_csv(path)
        # The filename is the authoritative gesture label: the per-person
        # directory is organized by gesture, and the in-file `taget` column is
        # occasionally mislabeled (e.g. ID8/waveright.csv has taget='waveleft').
        if file_label != gesture:
            print(f"  [loader] {person_id}/{gesture}.csv: taget='{file_label}' "
                  f"!= filename; using filename label '{gesture}'")
        label = gesture
        amp, phase = iq_to_amp_phase(iq)
        fs = target_fs
        if do_resample:
            amp, phase = resample_uniform(amp, phase, ts, target_fs)
        recordings.append(
            Recording(
                person=person_id,
                gesture_label=label,
                amp=amp,
                phase=phase,
                fs=fs,
                src_recording_id=f"{person_id}/{gesture}",
            )
        )
    return recordings
