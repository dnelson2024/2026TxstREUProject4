"""Widar3.0 raw CSI -> SHARP-style Doppler pipeline, condensed into ONE file.

Two tasks, selected with WIDAR_TASK (default 'har'):
  har   Widar3.0 gesture recordings (id-a-b-c-d-Rx.dat: user, gesture, torso
        location, face orientation, repetition, receiver r1-r6); label=gesture
  gait  Widar3.0/GaitID walk recordings (id-a-b-Rx.dat: user, track,
        repetition, receiver r1-r6); label=USER (11-way gait identification)

Both are raw Intel 5300 captures (1 Tx antenna x 3 Rx antennas x 30
subcarriers at ~1000 Hz) processed the same way SHARP steps 01-08 process
Nexmon data, as subcommands:

  manifest   -- index all .dat files under WIDAR_CSI (needed once, first)
  coherence  -- phase-coherence diagnostic for ONE file (--index N): prints
                packet-to-packet phase coherence R for raw and cross-antenna
                CSI (the CSI-Bench check that predicted sanitization failure)
  process    -- steps 01+02+03+04 for ONE file (--index N):
                01 signal preprocessing (per-packet amplitude normalize)
                02 phase sanitization H estimation (lasso over delay grid)
                03 signal reconstruction (offset-free phase rebuild)
                04 Doppler computation (STFT -> velocity spectrogram)
  dataset    -- steps 05+06: pad/crop traces to WIN_LENx100 windows and split
                train/val/test 70/15/15 stratified by gesture at INSTANCE
                level (all 6 receivers of a repetition stay together)
  plots      -- steps 07+08: example Doppler spectrograms per gesture

Differences from the CSI-Bench adaptation, driven by the hardware:
  * .dat parser is a numpy port of csi_tool_box read_bf_file/read_bfee/
    get_scaled_csi (record framing, 8-bit two's-complement bit unpacking,
    antenna permutation, SNR scaling).
  * frequency grid is the true non-uniform HT20 grouped-subcarrier layout
    (30 indices in [-28, 28], spacing 312.5 kHz) instead of a uniform grid.
  * Doppler STFT runs at the native ~1000 Hz: window 311 packets (~0.31 s,
    like SHARP's 31 at ~100 Hz), zero-padded FFT to 1000 bins, keep the
    central 100 (+-50 Hz at 1 Hz/bin -- gestures live within +-60 Hz),
    slide 10 packets (10 ms per slice, same time axis as SHARP).
  * gesture ids in filenames are SESSION-LOCAL; the per-date tables from the
    dataset README (GESTURES_BY_DATE below) map them to canonical names.

Paths (override via environment for LEAP2):
  WIDAR_CSI   raw .dat root, e.g. .../Widar3.0-HAR/20181130.../user5/...
  WIDAR_WORK  per-file Doppler outputs
  WIDAR_OUT   final dataset .npy
"""
import argparse
import csv
import glob
import json
import math as mt
import os
import re
import sys
import time

import numpy as np
import scipy
from scipy.fft import fft, fftshift
from scipy.signal.windows import hann

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE))  # repo root -> SHARP_Modified package
from SHARP_Modified.Python_code.optimization_utility import (  # noqa: E402
    build_T_matrix, lasso_regression_osqp_fast)

_DATA_ROOT = '/home/danelson/proj4-txstreu/2026TxstREUProject4-data'
# task -> (raw CSI dir, work dir, dataset dir, label column)
_TASKS = {
    'har': ('Widar3.0-HAR', 'widar_csi_work_har', 'widar_doppler_data_HAR',
            'gesture'),
    'gait': ('Widar3.0-Gait', 'widar_csi_work_gait', 'widar_doppler_data_Gait',
             'user'),
}
TASK = os.environ.get('WIDAR_TASK', 'har')
if TASK not in _TASKS:
    sys.exit("WIDAR_TASK must be 'har' or 'gait', got %r" % TASK)
LABEL_FIELD = _TASKS[TASK][3]
CSI_DIR = os.environ.get('WIDAR_CSI', os.path.join(_DATA_ROOT, _TASKS[TASK][0]))
WORK = os.environ.get('WIDAR_WORK', os.path.join(_DATA_ROOT, _TASKS[TASK][1]))
OUT = os.environ.get('WIDAR_OUT', os.path.join(_DATA_ROOT, _TASKS[TASK][2]))
MANIFEST = os.path.join(WORK, 'manifest.csv')

# Intel 5300 HT20 grouped subcarriers (802.11n-2009 table, Ng=2) -- NON-uniform
SUBC_IDX = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6,
                     -4, -2, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,
                     25, 27, 28])
FREQ_VECTOR = SUBC_IDX * 312.5e3
N_SUB, N_ANT = 30, 3

STFT_WIN = 311        # ~0.31 s at 1000 Hz (SHARP: 31 at ~100 Hz)
FFT_N = 1000          # zero-padded -> 1 Hz per bin
# central bins kept. har: +-50 Hz (SHARP output width 100; gestures < 60 Hz).
# gait: +-100 Hz -- walking torso is ~60 Hz (2v/lambda at 1.5 m/s, 5.825 GHz)
# and limb swings go higher, so +-50 would clip the signature.
# NOTE: must be identical between `process` and `dataset` runs of one task.
FFT_KEEP = int(os.environ.get('WIDAR_FFT_KEEP', 100 if TASK == 'har' else 200))
SLIDING = 10          # 10 ms per Doppler slice (SHARP: 1 packet at ~100 Hz)
FC = 5.825e9          # Widar3.0 carrier (channel 165); plot velocity axis only
DELTA_V = 2.99792458e8 / FC  # m/s per 1-Hz bin (SHARP convention v = fD*lambda)
NOISE_LEV = -1.2      # floor at 10^-1.2 of per-slice max (SHARP: -1.2)
TRIM = 100            # packets dropped at each end (SHARP: 100 at 10x the Tc)
WIN_LEN = 340         # dataset window length; short traces noise-floor padded
SPLIT_SEED, SPLIT_FRACS = 42, (0.70, 0.15, 0.15)

# Doppler source. 'raw' = STFT the coherent raw CSI directly. Intel 5300 phase
# here is already coherent (raw R~0.74, cross-antenna R~1.0), so the gesture
# Doppler is common-mode; the SHARP per-packet re-referencing (stage02/03) and
# the antenna-conjugate trick both CANCEL it -> dead-flat 0 Hz spectrograms.
# Verified on user10-1-1-1-1-r1.dat: raw shows the moving Push&Pull line,
# sanitized is flat. 'sanitized' keeps the old SHARP path (correct for Nexmon).
REPRESENTATION = os.environ.get('WIDAR_REPR', 'raw')

# Dataset README: gesture ids are numbered per collection date (and for three
# dates per user). Table transcribed from README.pdf pages 1-3.
_G6V = {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Draw-O(V)',
        5: 'Draw-Zigzag(V)', 6: 'Draw-N(V)'}
_GD = {1: 'Draw-1', 2: 'Draw-2', 3: 'Draw-3', 4: 'Draw-4', 5: 'Draw-5',
       6: 'Draw-6', 7: 'Draw-7', 8: 'Draw-8', 9: 'Draw-9', 10: 'Draw-10'}
_G6H = {1: 'Slide', 2: 'Draw-O(H)', 3: 'Draw-Zigzag(H)', 4: 'Draw-N(H)',
        5: 'Draw-Triangle(H)', 6: 'Draw-Rectangle(H)'}
_G9 = {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide', 5: 'Draw-O(H)',
       6: 'Draw-Zigzag(H)', 7: 'Draw-N(H)', 8: 'Draw-Triangle(H)',
       9: 'Draw-Rectangle(H)'}
_G6HZ = {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide', 5: 'Draw-O(H)',
         6: 'Draw-Zigzag(H)'}
GESTURES_BY_DATE = {
    '20181109': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide',
                 5: 'Draw-Zigzag(V)', 6: 'Draw-N(V)'},
    '20181112': _GD, '20181115': _G6V, '20181116': _GD, '20181117': _G6V,
    '20181118': _G6V, '20181121': _G6H, '20181127': _G6H,
    '20181128': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Draw-O(H)',
                 5: 'Draw-Zigzag(H)', 6: 'Draw-N(H)'},
    '20181130': _G9, '20181204': _G9,
    '20181205': {'user2': {1: 'Draw-O(H)', 2: 'Draw-Zigzag(H)', 3: 'Draw-N(H)',
                           4: 'Draw-Triangle(H)', 5: 'Draw-Rectangle(H)'},
                 'user3': _G6H},
    '20181208': {'user2': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide'},
                 'user3': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap'}},
    '20181209': {'user2': {1: 'Push&Pull'}, 'user6': _G6HZ},
    '20181211': _G6HZ,
}
ROOM_BY_DATE = {'20181109': 1, '20181112': 1, '20181115': 1, '20181116': 1,
                '20181117': 2, '20181118': 2, '20181121': 1, '20181127': 2,
                '20181128': 2, '20181130': 1, '20181204': 2, '20181205': 2,
                '20181208': 2, '20181209': 2, '20181211': 3}

# Gait (README_Gait.pdf): Room#2 = the 20190719 sessions of user1/user2/user11,
# everything else Room#1. The *_sup dirs are supplementary Room#1 sessions with
# extra tracks 5-6 (main sessions use tracks 1-4). For repetition no., odd =
# forward direction along the track, even = reverse.
GAIT_ROOM2 = {('20190719', 'user1'), ('20190719', 'user2'),
              ('20190719', 'user11')}


def rec_tag(row):
    # Path-based (NOT index-based): manifest indices shift whenever new dates
    # are added, which would silently re-attach cached npz to the wrong files.
    return re.sub(r'[^A-Za-z0-9._-]', '_', row['path'])[:150]


def read_manifest():
    with open(MANIFEST) as fp:
        return list(csv.DictReader(fp))


# --------------------------------------------------------------------------
# Intel 5300 .dat parser (numpy port of csi_tool_box read_bf_file/read_bfee/
# get_scaled_csi, (c) Daniel Halperin)
# --------------------------------------------------------------------------
def read_bf_file(path):
    """Returns (csi, ts): csi (T, N_ANT, N_SUB) complex scaled to sqrt(SNR),
    ts (T,) microsecond NIC timestamps. Non-3x1 or malformed records are
    dropped; empty files (see README Bug Notice) return (0, ...) arrays."""
    raw = np.fromfile(path, dtype=np.uint8)
    payloads, headers = [], []
    cur, total = 0, len(raw)
    while cur < total - 3:
        field_len = (int(raw[cur]) << 8) + int(raw[cur + 1])   # u16 big-endian
        code = raw[cur + 2]
        cur += 3
        if cur + field_len - 1 > total:
            break
        if code == 187:
            rec = raw[cur:cur + field_len - 1]
            if len(rec) >= 20 and rec[8] == N_ANT and rec[9] == 1:
                calc_len = (30 * (int(rec[8]) * int(rec[9]) * 8 * 2 + 3) + 7) // 8
                if int(rec[16]) + (int(rec[17]) << 8) == calc_len \
                        and len(rec) >= 20 + calc_len:
                    headers.append(rec[:20])
                    payloads.append(rec[20:20 + calc_len])
        cur += field_len - 1
    if not payloads:
        return (np.zeros((0, N_ANT, N_SUB), complex), np.zeros(0))
    H = np.stack(headers).astype(np.int64)     # (T, 20)
    P = np.stack(payloads)                     # (T, calc_len) uint8

    # bit offsets: per subcarrier index += 3, then 16 bits per (re, im) pair
    bit = 3 + 16 * N_ANT * np.arange(N_SUB)[:, None] + \
        16 * np.arange(N_ANT)[None, :] + \
        np.array(0)                            # (N_SUB, N_ANT) start bit of re
    bit = (bit[..., None] + np.array([0, 8])).reshape(-1)   # ... then im
    # cumulative +3 per subcarrier: index starts at 3 and gains 3 each row
    bit = bit + 3 * np.repeat(np.arange(N_SUB), N_ANT * 2)
    idx, rem = bit // 8, bit % 8
    vals = ((P[:, idx] >> rem) |
            (P[:, idx + 1].astype(np.uint16) << (8 - rem))).astype(np.uint8)
    vals = vals.view(np.int8).astype(np.float64)            # two's complement
    vals = vals.reshape(-1, N_SUB, N_ANT, 2)
    csi = (vals[..., 0] + 1j * vals[..., 1]).transpose(0, 2, 1)  # (T, ANT, SUB)

    # antenna permutation (valid perms only, like read_bf_file.m)
    perm = np.stack([H[:, 15] & 0x3, (H[:, 15] >> 2) & 0x3,
                     (H[:, 15] >> 4) & 0x3], axis=1)
    ok = perm.sum(axis=1) == 3                 # 0+1+2: a real permutation
    out = csi.copy()
    rows = np.arange(len(csi))[ok]
    out[rows[:, None], perm[ok]] = csi[ok]

    # get_scaled_csi: scale to sqrt(SNR) units
    dbinv = lambda x: 10.0 ** (x / 10.0)       # noqa: E731
    rssi_mag = sum(np.where(H[:, 10 + i] != 0, dbinv(H[:, 10 + i]), 0)
                   for i in range(3))
    rssi_pwr = dbinv(10 * np.log10(np.maximum(rssi_mag, 1e-12)) - 44 - H[:, 14])
    csi_pwr = np.abs(out).reshape(len(out), -1).__pow__(2).sum(axis=1)
    scale = rssi_pwr / np.maximum(csi_pwr / 30, 1e-12)
    noise_db = np.where(H[:, 13].astype(np.int8) == -127, -92,
                        H[:, 13].astype(np.int8))
    total_noise = dbinv(noise_db) + scale * N_ANT
    out = out * np.sqrt(scale / total_noise)[:, None, None]

    ts = (H[:, 0] + (H[:, 1] << 8) + (H[:, 2] << 16) + (H[:, 3] << 24)).astype(float)
    return out, ts


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def gesture_name(date, user, gid):
    table = GESTURES_BY_DATE.get(date)
    if isinstance(table, dict) and user in table:
        table = table[user]
    return table.get(gid) if isinstance(table, dict) else None


def _manifest_row_har(basename, date, rel, n_rows):
    m = re.match(r'(user\d+)-(\d+)-(\d+)-(\d+)-(\d+)-[rR](\d)\.dat$', basename)
    if not m or date not in GESTURES_BY_DATE:
        return None
    user, gid = m.group(1), int(m.group(2))
    name = gesture_name(date, user, gid)
    if name is None:
        return None
    return {'index': n_rows, 'path': rel, 'date': date,
            'room': ROOM_BY_DATE[date], 'user': user,
            'gesture': name, 'torso_location': m.group(3),
            'face_orientation': m.group(4), 'repetition': m.group(5),
            'receiver': 'r' + m.group(6),
            'instance': '%s-%s-%s' % (date, user, '-'.join(m.group(2, 3, 4, 5)))}


def _manifest_row_gait(basename, date, rel, n_rows):
    m = re.match(r'(user\d+)-(\d+)-(\d+)-[rR](\d)\.dat$', basename)
    if not m or not date:
        return None
    user, track, rep = m.group(1), m.group(2), m.group(3)
    return {'index': n_rows, 'path': rel, 'date': date,
            'room': 2 if (date, user) in GAIT_ROOM2 else 1, 'user': user,
            'track': track,
            'direction': 'forward' if int(rep) % 2 else 'reverse',
            'repetition': rep, 'receiver': 'r' + m.group(4),
            'instance': '%s-%s-%s-%s' % (date, user, track, rep)}


def cmd_manifest(_args):
    os.makedirs(WORK, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CSI_DIR, '**', '*.dat'), recursive=True))
    make_row = _manifest_row_gait if TASK == 'gait' else _manifest_row_har
    rows, unknown, unparsed = [], 0, []
    for f in files:
        rel = os.path.relpath(f, CSI_DIR)
        date = next((p[:8] for p in rel.split(os.sep) if re.match(r'^20\d{6}', p)), '')
        row = make_row(os.path.basename(f), date, rel, len(rows))
        if row is None:
            unknown += 1
            if len(unparsed) < 5:
                unparsed.append(rel)
            continue
        rows.append(row)
    if not rows:
        sys.exit('manifest: NO usable .dat files under %s (task %s)\n'
                 '  .dat files found: %d, rejected by name/date parse: %d\n'
                 '  first rejected paths: %s\n'
                 '  expected layout like  %s\n'
                 '  (a path component must start with the 8-digit date; check '
                 'that the zip downloaded fully and unzipped there)'
                 % (CSI_DIR, TASK, len(files), unknown, unparsed or '(none)',
                    '20190719/user10/user10-1-12-r4.dat' if TASK == 'gait'
                    else '20181130/.../user5-1-1-1-1-r1.dat'))
    with open(MANIFEST, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    labels = sorted({r[LABEL_FIELD] for r in rows})
    print('manifest[%s]: %d files, %d instances, %d %ss (%d files skipped) -> %s'
          % (TASK, len(rows), len({r['instance'] for r in rows}), len(labels),
             LABEL_FIELD, unknown, MANIFEST))
    for g in labels:
        print('  %-22s %5d files' % (g, sum(r[LABEL_FIELD] == g for r in rows)))


# --------------------------------------------------------------------------
# step 01: signal preprocessing (per-packet amplitude normalization)
# --------------------------------------------------------------------------
def stage01_signal(dat_path, limit_packets=0):
    csi, ts = read_bf_file(dat_path)
    if limit_packets:
        csi = csi[:limit_packets]
    keep = np.abs(csi).sum(axis=(1, 2)) > 0
    csi = csi[keep]
    signal_complete = csi.transpose(2, 0, 1).astype(complex)  # (n_sub, T, ant)
    mean_amp = np.mean(np.abs(signal_complete), axis=0, keepdims=True)
    return signal_complete / np.maximum(mean_amp, 1e-12), ts[keep]


# --------------------------------------------------------------------------
# step 02: phase sanitization H estimation (SHARP lasso, HT20 grid)
# --------------------------------------------------------------------------
def stage02_hest(signal_complete):
    delta_t, delta_t_refined = 1e-7, 5e-9
    range_refined_up, range_refined_down = 2.5e-7, 2e-7
    t_min, t_max = -3e-7, 5e-7
    T_matrix, time_matrix = build_T_matrix(FREQ_VECTOR, delta_t, t_min, t_max)

    select_subcarriers = np.arange(0, N_SUB, 1)
    row_T, col_T = T_matrix.shape
    m, n = 2 * row_T, 2 * col_T
    In, Im = scipy.sparse.eye(n), scipy.sparse.eye(m)
    On = scipy.sparse.csc_matrix((n, n))
    Onm = scipy.sparse.csc_matrix((n, m))
    P = scipy.sparse.block_diag([On, Im, On], format='csc')
    q = np.zeros(2 * n + m)
    A2 = scipy.sparse.hstack([In, Onm, -In])
    A3 = scipy.sparse.hstack([In, Onm, In])
    ones_n, zeros_n, zeros_nm = np.ones(n), np.zeros(n), np.zeros(n + m)

    n_time, n_ant = signal_complete.shape[1], signal_complete.shape[2]
    sanitized = np.zeros((N_SUB, n_time, n_ant), dtype=complex)
    for stream in range(n_ant):
        for ts in range(n_time):
            signal_time = signal_complete[:, ts, stream]
            r = lasso_regression_osqp_fast(signal_time, T_matrix, select_subcarriers,
                                           row_T, col_T, Im, Onm, P, q, A2, A3,
                                           ones_n, zeros_n, zeros_nm)
            time_max = time_matrix[np.argmax(abs(r))]

            T_ref, time_ref = build_T_matrix(FREQ_VECTOR, delta_t_refined,
                                             max(time_max - range_refined_down, t_min),
                                             min(time_max + range_refined_up, t_max))
            col_ref = T_ref.shape[1]
            n_ref = 2 * col_ref
            In_r = scipy.sparse.eye(n_ref)
            On_r = scipy.sparse.csc_matrix((n_ref, n_ref))
            Onm_r = scipy.sparse.csc_matrix((n_ref, m))
            P_r = scipy.sparse.block_diag([On_r, Im, On_r], format='csc')
            q_r = np.zeros(2 * n_ref + m)
            A2_r = scipy.sparse.hstack([In_r, Onm_r, -In_r])
            A3_r = scipy.sparse.hstack([In_r, Onm_r, In_r])
            r_ref = lasso_regression_osqp_fast(signal_time, T_ref, select_subcarriers,
                                               row_T, col_ref, Im, Onm_r, P_r, q_r,
                                               A2_r, A3_r, np.ones(n_ref),
                                               np.zeros(n_ref), np.zeros(n_ref + m))
            pos_max = np.argmax(abs(r_ref))
            Tr = np.multiply(T_ref, r_ref)
            # phase-reference every path to the strongest one -> removes the
            # common (CFO/PLL/SFO) phase offsets; this IS the sanitization
            Trr = np.multiply(Tr, np.conj(Tr[:, pos_max:pos_max + 1]))
            sanitized[:, ts, stream] = np.sum(Trr, axis=1)
    return sanitized


# --------------------------------------------------------------------------
# step 03: signal reconstruction (offset-free phase rebuild) -- same as SHARP
# --------------------------------------------------------------------------
def stage03_reconstruct(H_est):                       # (n_sub, T) one antenna
    n_sub, n_time = H_est.shape
    amp = np.abs(H_est).T                             # (T, n_sub)

    phase = np.unwrap(np.angle(H_est), axis=0)        # unwrap along subcarriers
    for tidx in range(1, n_time):
        idx_prec, stop = -1, False
        while not stop:
            phase_err = phase[:, tidx] - phase[:, tidx - 1]
            diff_err = np.diff(phase_err)
            up = np.argwhere(diff_err > 0.9 * mt.pi)[:, 0]
            down = np.argwhere(diff_err < -0.9 * mt.pi)[:, 0]
            if up.shape[0] > 0:
                if up[0] == idx_prec:
                    stop = True
                else:
                    phase[up[0] + 1:, tidx] -= 2 * mt.pi
                    idx_prec = up[0]
            elif down.shape[0] > 0:
                if down[0] == idx_prec:
                    stop = True
                else:
                    phase[down[0] + 1:, tidx] += 2 * mt.pi
                    idx_prec = down[0]
            else:
                stop = True
    ones_vector = np.ones((2, n_sub))
    ones_vector[1, :] = np.arange(n_sub)
    for tidx in range(1, n_time - 1):
        error = phase[:, tidx:tidx + 1] - phase[:, tidx - 1:tidx]
        coeff = np.linalg.lstsq(ones_vector.T, error, rcond=None)[0]
        phase[:, tidx] -= np.dot(ones_vector.T, coeff)[:, 0]
    return amp, phase.T                               # both (T, n_sub)


# --------------------------------------------------------------------------
# step 04: Doppler computation (STFT at 1000 Hz -> +-50 Hz spectrogram)
# --------------------------------------------------------------------------
def doppler_stft(csi_complex):
    """STFT a complex (T, n_sub) CSI stream -> (n_slices, FFT_KEEP) power map."""
    window = np.expand_dims(hann(STFT_WIN), axis=-1)
    profiles = []
    for i in range(0, csi_complex.shape[0] - STFT_WIN, SLIDING):
        cut = np.nan_to_num(csi_complex[i:i + STFT_WIN, :]) * window
        prof = fftshift(fft(cut, n=FFT_N, axis=0), axes=0)
        prof = prof[FFT_N // 2 - FFT_KEEP // 2:FFT_N // 2 + FFT_KEEP // 2]
        profiles.append(np.sum(np.abs(prof * np.conj(prof)), axis=1))
    d = np.asarray(profiles)                          # (n_bins, FFT_KEEP)
    d = d / np.max(d, axis=1, keepdims=True)
    d[d < mt.pow(10, NOISE_LEV)] = mt.pow(10, NOISE_LEV)
    return d.astype(np.float32)


def stage04_doppler(amp, phase):                      # SHARP-sanitized path
    amp = amp / np.mean(amp, axis=1, keepdims=True)
    return doppler_stft(amp * np.exp(1j * phase))


def stage_raw_doppler(signal_complete):
    """'raw' path: Doppler straight from the coherent (stage01-normalized) CSI,
    with NO per-packet phase re-referencing -- that cancels the common-mode
    gesture Doppler on this data. (n_sub, T, ant) -> (ant, n_slices, FFT_KEEP)."""
    dopplers = [doppler_stft(signal_complete[:, :, stream].T)
                for stream in range(signal_complete.shape[2])]
    return np.stack(dopplers)


# --------------------------------------------------------------------------
# coherence diagnostic: is per-packet phase workable at all? (On CSI-Bench,
# raw R=0.003 predicted the sanitization failure before any training did.)
# --------------------------------------------------------------------------
def cmd_coherence(args):
    row = read_manifest()[args.index]
    signal, ts = stage01_signal(os.path.join(CSI_DIR, row['path']))
    print('%s: %s %s %s, %d packets' % (row['path'], row['user'],
          row[LABEL_FIELD], row['receiver'], signal.shape[1]))
    if signal.shape[1] < 10:
        print('too short for diagnostics')
        return
    dt = np.diff(ts)
    dt = dt[(dt > 0) & (dt < 1e5)]
    if len(dt):
        print('median packet interval: %.2f ms (nominal 1.0)' % (np.median(dt) / 1e3))
    for name, h in [('raw           ', signal[:, :, 0]),
                    ('cross-antenna ', signal[:, :, 0] * np.conj(signal[:, :, 1]))]:
        num = np.abs(np.mean(h[:, 1:] * np.conj(h[:, :-1]), axis=1))
        den = np.mean(np.abs(h[:, 1:]) * np.abs(h[:, :-1]), axis=1)
        r = float(np.mean(num / np.maximum(den, 1e-12)))
        print('phase coherence R %s = %.3f  %s' % (name, r,
              '(coherent)' if r > 0.5 else '(RANDOMIZED -- sanitization will fail)'))


# --------------------------------------------------------------------------
# process = 01 -> 02 -> 03 -> 04 for one manifest entry
# --------------------------------------------------------------------------
def cmd_process(args):
    rows = read_manifest()
    if args.index >= len(rows):
        print('index %d beyond manifest (%d entries) -- nothing to do' % (args.index, len(rows)))
        return
    row = rows[args.index]
    out_file = os.path.join(WORK, rec_tag(row) + '_doppler.npz')
    if os.path.exists(out_file):
        print('already processed: %s' % out_file)
        return
    t0 = time.time()
    cond = (' loc%s ori%s' % (row['torso_location'], row['face_orientation'])
            if TASK == 'har' else
            ' track%s %s' % (row['track'], row['direction']))
    print('[%s] %s (%s %s%s rep%s %s)' % (rec_tag(row), row['path'],
          row['user'], row[LABEL_FIELD], cond, row['repetition'],
          row['receiver']))

    signal, _ts = stage01_signal(os.path.join(CSI_DIR, row['path']), args.limit_packets)
    signal = signal[:, TRIM:max(0, signal.shape[1] - TRIM), :]
    if signal.shape[1] <= STFT_WIN + SLIDING:
        print('too short (%d packets after trim) -- skipping' % signal.shape[1])
        return
    print('  01 signal: %s  (%.1f s)' % (signal.shape, time.time() - t0), flush=True)

    if REPRESENTATION == 'raw':
        doppler = stage_raw_doppler(signal)           # (ant, n_bins, FFT_KEEP)
        print('  raw doppler: %s (%.1f s)' % (doppler.shape, time.time() - t0), flush=True)
    else:
        sanitized = stage02_hest(signal)
        print('  02 H estimation done (%.1f min)' % ((time.time() - t0) / 60), flush=True)
        dopplers = []
        for stream in range(signal.shape[2]):
            amp, phase = stage03_reconstruct(sanitized[:, :, stream])
            dopplers.append(stage04_doppler(amp, phase))
        doppler = np.stack(dopplers)                  # (3, n_bins, FFT_KEEP)
        print('  03+04 doppler: %s (total %.1f min)' % (doppler.shape, (time.time() - t0) / 60))

    os.makedirs(WORK, exist_ok=True)
    tmp = out_file + '.tmp.npz'
    np.savez_compressed(tmp, doppler=doppler)
    os.replace(tmp, out_file)
    with open(os.path.join(WORK, rec_tag(row) + '_meta.json'), 'w') as fp:
        json.dump({'n_bins': int(doppler.shape[1]), **row}, fp)
    print('saved %s' % out_file)


# --------------------------------------------------------------------------
# steps 05+06: pad/crop to WIN_LEN + instance-level stratified splits -> .npy
# --------------------------------------------------------------------------
def assign_splits(rows):
    rng = np.random.default_rng(SPLIT_SEED)
    by_label = {}
    for r in rows:
        by_label.setdefault(r[LABEL_FIELD], set()).add(r['instance'])
    assignment = {}
    for g in sorted(by_label):
        instances = sorted(by_label[g])
        rng.shuffle(instances)
        n_tr = int(round(SPLIT_FRACS[0] * len(instances)))
        n_va = int(round(SPLIT_FRACS[1] * len(instances)))
        for i, inst in enumerate(instances):
            assignment[inst] = ('train' if i < n_tr else
                                'val' if i < n_tr + n_va else 'test')
    return assignment


def cmd_dataset(_args):
    rows = read_manifest()
    assignment = assign_splits(rows)
    labels = sorted({r[LABEL_FIELD] for r in rows})
    label_idx = {g: i for i, g in enumerate(labels)}
    floor = mt.pow(10, NOISE_LEV)

    done = [r for r in rows
            if os.path.exists(os.path.join(WORK, rec_tag(r) + '_doppler.npz'))]
    print('processed files: %d / %d' % (len(done), len(rows)))
    counts = {}
    for r in done:
        counts[assignment[r['instance']]] = counts.get(assignment[r['instance']], 0) + 1

    os.makedirs(OUT, exist_ok=True)
    xmaps, ys, metas, filled = {}, {}, {}, {}
    for split, n in counts.items():
        xmaps[split] = np.lib.format.open_memmap(
            os.path.join(OUT, '%s_X.npy' % split), mode='w+',
            dtype=np.float16, shape=(n, N_ANT, WIN_LEN, FFT_KEEP))
        ys[split], metas[split], filled[split] = np.zeros(n, np.int64), [], 0

    for r in done:
        doppler = np.load(os.path.join(WORK, rec_tag(r) + '_doppler.npz'))['doppler']
        n_bins = doppler.shape[1]
        win = np.full((N_ANT, WIN_LEN, FFT_KEEP), floor, np.float32)
        if n_bins >= WIN_LEN:                         # center-crop
            s = (n_bins - WIN_LEN) // 2
            win[:] = doppler[:, s:s + WIN_LEN, :]
        else:                                         # center-pad on the floor
            s = (WIN_LEN - n_bins) // 2
            win[:, s:s + n_bins, :] = doppler
        split = assignment[r['instance']]
        i = filled[split]
        xmaps[split][i] = win.astype(np.float16)
        ys[split][i] = label_idx[r[LABEL_FIELD]]
        meta_fields = [k for k in r if k not in ('index', 'path')]
        metas[split].append([rec_tag(r)] + [r[k] for k in meta_fields] + [n_bins])
        filled[split] += 1

    with open(os.path.join(OUT, 'labels.json'), 'w') as fp:
        json.dump(labels, fp)
    meta_header = (['recording']
                   + [k for k in rows[0] if k not in ('index', 'path')]
                   + ['n_bins'])
    for split in counts:
        assert filled[split] == counts[split]
        xmaps[split].flush()
        np.save(os.path.join(OUT, '%s_y.npy' % split), ys[split])
        with open(os.path.join(OUT, '%s_meta.csv' % split), 'w', newline='') as fp:
            w = csv.writer(fp)
            w.writerow(meta_header)
            w.writerows(metas[split])
        print('%-6s X=%s' % (split, xmaps[split].shape))
    print('done -> %s' % OUT)


# --------------------------------------------------------------------------
# steps 07+08: example Doppler plots per gesture
# --------------------------------------------------------------------------
def cmd_plots(_args):
    # SHARP paper style (plots_utility.plt_doppler_antennas / slurm config.sh):
    # viridis, dB dynamic range [NOISE_LEV*10, 0] with colorbar ticks
    # [-12, -8, -4, 0], y axis in velocity (v = f_D * lambda, SHARP convention).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_manifest()
    plot_dir = os.path.join(WORK, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    v_max = FFT_KEEP / 2 * DELTA_V
    for g in sorted({r[LABEL_FIELD] for r in rows}):
        row = next((r for r in rows if r[LABEL_FIELD] == g and
                    os.path.exists(os.path.join(WORK, rec_tag(r) + '_doppler.npz'))), None)
        if row is None:
            print('no processed file yet for %s' % g)
            continue
        d = np.load(os.path.join(WORK, rec_tag(row) + '_doppler.npz'))['doppler']
        t_max = d.shape[1] * SLIDING / 1000.0     # s (1000 Hz packet rate)
        fig, axes = plt.subplots(N_ANT, 1, figsize=(9, 8), sharex=True)
        for ant in range(N_ANT):
            im = axes[ant].imshow(10 * np.log10(d[ant].T), aspect='auto',
                                  origin='lower', cmap='viridis',
                                  vmin=10 * NOISE_LEV, vmax=0,
                                  extent=[0, t_max, -v_max, v_max])
            axes[ant].set_ylabel('antenna %d\nvelocity [m/s]' % ant)
            cbar = fig.colorbar(im, ax=axes[ant], ticks=[-12, -8, -4, 0])
            cbar.ax.set_ylabel('power [dB]')
        axes[-1].set_xlabel('time [s]')
        fig.suptitle('Widar3.0 %s Doppler -- %s (%s %s %s)' % (
            TASK.upper(), g, row['user'], row['date'], row['receiver']))
        fig.tight_layout()
        safe = g.replace('&', 'and').replace('(', '').replace(')', '')
        fig.savefig(os.path.join(plot_dir, 'doppler_%s.png' % safe), dpi=120)
        plt.close(fig)
        print('plotted %s' % g)
    print('plots -> %s' % plot_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('manifest')
    for name in ('process', 'coherence'):
        p = sub.add_parser(name)
        p.add_argument('--index', type=int, required=True)
        if name == 'process':
            p.add_argument('--limit_packets', type=int, default=0,
                           help='debug: only process the first N packets')
    sub.add_parser('dataset')
    sub.add_parser('plots')
    args = parser.parse_args()
    {'manifest': cmd_manifest, 'coherence': cmd_coherence, 'process': cmd_process,
     'dataset': cmd_dataset, 'plots': cmd_plots}[args.cmd](args)
