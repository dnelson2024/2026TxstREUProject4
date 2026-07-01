# CSI Movement Classification — Implementation Plan

## Context

This is Texas State REU Project 4 (smart homes for ASD using IoT + AI). The dataset
contains ESP32 WiFi **Channel State Information (CSI)** recordings of human gestures, and
the goal is to classify which gesture is being performed from the CSI signal. Currently the
repo has only the raw dataset and a stub notebook (`test1.ipynb`) — no pipeline exists.

This plan builds a reproducible Python pipeline that learns gesture representations with
**self-supervised SimCLR contrastive pretraining**, evaluates them with a **supervised
linear probe**, and compares three encoder architectures (MLP, CNN, ViT).

### Confirmed scope decisions
- **Dynamic gestures only** — 6 classes: `applause, circleclockwise, frontandafter, leftandright, upanddown, waveright`.
- **Per-person models** — train + evaluate independently for each of the 8 people (ID1–ID8). The full run = 8 people × 3 encoders = 24 runs.
- **SimCLR contrastive + linear probe** — contrastive pretrain (no labels) → freeze encoder → supervised linear probe on true labels → metrics.

### Verified data facts (checked directly, correcting the original brief)
- CSVs have **28 columns**, not 131. Label is column **`taget`** (misspelled in the data — use literal string).
- CSI is in the `data` column as a **JSON-style array of 104 ints = 52 subcarriers × interleaved I/Q** (`I0,Q0,I1,Q1,...`). Amplitude = `hypot(I,Q)`, phase = `arctan2(Q,I)`.
- Effective sampling rate is **~82 Hz and non-uniform** (nominal 100 Hz, but ESP32 drops packets). **Must resample to a uniform grid before filtering/STFT.**
- Each dynamic file ≈ 5000 rows (~60 s), one gesture per file.
- **CSIKit is NOT needed** — data is already raw CSI in CSV; parse directly with pandas/numpy.
- Env: `.conda/bin/python` = Python 3.12.13 (only numpy present).

---

## Key design resolutions

### Tensor shape: `[2, 52, 250]` (channels × subcarrier × time)
The original `[2, F, T]` from per-subcarrier STFT is under-specified (the 52-subcarrier axis has nowhere to go). **Default approach (Option A):** treat the sample as a 2-channel image — channel 0 = amplitude, channel 1 = phase, F=52 subcarriers, T=250 time steps. No STFT in the default path; the encoders act on the (subcarrier × time) image. Keep `use_stft` as a config flag for a later experiment (would produce `[2, 52, 33, 12]`, requiring a documented collapse).

### Pipeline order: clean full series → window → (optional STFT)
Filtering (Hampel/Butterworth/unwrap/z-score) needs the full recording for stable edges, so it runs **before** windowing. Each window then becomes a fixed `[2, 52, 250]` tensor. (Window-first-then-STFT, not STFT-the-whole-recording, to keep window boundaries well-defined.)

### Hungarian alignment is NOT used in the default path
Since the linear probe is **supervised** (trained on true labels), its outputs already live in the true-label space — Hungarian is unnecessary and, applied to test, can inflate metrics. **Default `eval_mode="linear_probe"` skips Hungarian.** Hungarian is retained only for an optional `eval_mode="cluster_hungarian"` path (k-means on frozen embeddings), where the cluster→label mapping is fit on **train** and frozen before applying to test.

### Leakage prevention (critical)
50%-overlap windows mean random splitting would leak shared raw-time content into test. **Split temporally per file:** first 80% of a recording's windows → train, next 10% → val, last 10% → test, **dropping windows that straddle a boundary** (≥1-window guard gap). All 6 classes appear in every split because each gesture is its own file. Z-score and Hampel stats are **fit on train only**, applied to val/test.

---

## File structure

```
csi_pipeline/
├── config.py                  # single CONFIG dict (defaults below)
├── data/
│   ├── loader.py              # parse CSV, I/Q→amp/phase, resample
│   ├── preprocess.py          # Hampel, Butterworth, phase sanitize, z-score (fit/transform)
│   └── windowing.py           # sliding windows + temporal split + tensor builder
├── models/
│   ├── heads.py               # ProjectionHead, LinearProbe
│   ├── mlp.py                 # MLPEncoder
│   ├── cnn.py                 # CNNEncoder
│   └── vit.py                 # ViTEncoder
├── training/
│   ├── contrastive.py         # NT-Xent loss, SimCLR augmentations
│   ├── trainer.py             # pretrain_contrastive, freeze, train_linear_probe
│   └── evaluate.py            # predict, metrics, (optional) hungarian_align
├── plots/visualize.py         # confusion, t-SNE, training curves, results bars
└── run_experiment.py          # orchestrates 8 people × 3 encoders, aggregates, plots
```

---

## Module responsibilities

### `data/loader.py`
- `parse_csi_csv(path) -> (iq[N,104], label)` — read CSV, parse `data` column, validate each row is length 104 (drop/log malformed), label from `taget`.
- `iq_to_amp_phase(iq) -> (amp[N,52], phase[N,52])` — `I=iq[:,0::2]`, `Q=iq[:,1::2]`.
- `resample_uniform(amp, phase, timestamps, target_fs) -> (amp_u, phase_u)` — linear interp onto uniform 100 Hz grid. **Unwrap phase before interpolating it** (interpolating across ±π jumps is invalid).
- `load_person_recordings(person_id, data_root) -> list[Recording]` — all 6 dynamic CSVs for a person. `Recording = {person, gesture_label, amp, phase, fs}`.

### `data/preprocess.py` (full-recording arrays; no windowing)
- `hampel_filter(x, window, n_sigmas)` — per-subcarrier rolling median+MAD outlier replacement.
- `butter_lowpass(x, cutoff_hz, fs, order)` — `scipy.signal.butter` + `filtfilt` (zero-phase), per column. cutoff 30 Hz < Nyquist 50 Hz: valid.
- `phase_sanitize(phase)` — per-timestep linear-fit removal across 52 subcarriers (kill CFO/SFO slope) → `np.unwrap` along time → `savgol_filter`.
- `zscore_per_subcarrier(x, stats=None) -> (x, stats)` — **fit/transform separated for leakage control** (stats=None computes & returns; else applies given stats).

### `data/windowing.py`
- `sliding_windows(amp, phase, label, win=250, stride=125) -> list[Window]` — each carries `src_recording_id` + `window_index` (needed for leakage-safe split).
- `temporal_split(windows, train=.8, val=.1, test=.1, guard=1)` — contiguous per-file split with guard-gap drop; stratified by gesture.
- `windows_to_tensor(window) -> np.ndarray[2,52,250]` — transpose each `[250,52]→[52,250]`, stack [amp, phase].

### `models/` — each `Encoder: [B,2,52,250] -> [B, embed_dim=128]`
- `mlp.py`: flatten (2·52·250=26000) → 3× (Linear→BN→ReLU→Dropout), hidden 256.
- `cnn.py`: 3× (Conv2d→BN→ReLU→MaxPool), channels [16,32,64] → AdaptiveAvgPool2d(1) → embed_dim.
- `vit.py`: patch embed Conv2d(stride=patch=(13,25) → 4×10=40 patches) → +CLS → learned pos enc → TransformerEncoder(dim 128, depth 4, heads 4) → CLS token.
- `heads.py`: `ProjectionHead(in,hidden,out)` (2-layer MLP), `LinearProbe(in, n_classes=6)`.

### `training/contrastive.py`
- `nt_xent_loss(z[2B,D], temperature)` — **L2-normalize embeddings**, similarity `z@z.T/temp`, **mask diagonal**, positive of i is i+B (and vice versa), cross-entropy over rows averaged across 2B. Use fp32 (or −1e9 mask not −inf under AMP).
- `augment(batch) -> (view1, view2)` — Gaussian noise, contiguous time-masking, subcarrier-masking, small time-shift, amplitude scaling. **Phase channel augmented gentler than amplitude.**

### `training/trainer.py`
- `pretrain_contrastive(encoder, proj_head, train_loader, cfg)` — SimCLR loop, train windows only, no labels, `drop_last=True`.
- `freeze(encoder)`; `train_linear_probe(frozen_encoder, probe, labeled_train, val, cfg)` — CE on frozen embeddings.
- `extract_embeddings(encoder, loader) -> (Z, y)` — for t-SNE / cluster path.

### `training/evaluate.py`
- `predict(encoder, probe, test_loader) -> (y_pred, y_true)`.
- `metrics(y_true, y_pred, class_names)` — per-class + macro precision/recall/F1, accuracy, confusion matrix (via `sklearn.metrics`).
- `hungarian_align(...)` — **only** for `cluster_hungarian` mode; fit mapping on train, freeze, apply to test (`scipy.optimize.linear_sum_assignment`).

### `run_experiment.py`
Seeds everything. Loops `for person: for encoder:` → load → resample → preprocess (fit stats on train) → window → temporal split → pretrain → freeze → probe → evaluate → collect tidy records. Aggregates and writes `results/` table + plots.

---

## CONFIG defaults (`config.py`)

```python
CONFIG = {
  "seed": 42,
  "data_root": "dataset/dynamic",
  "person_ids": ["ID1","ID2","ID3","ID4","ID5","ID6","ID7","ID8"],
  "classes": ["applause","circleclockwise","frontandafter",
              "leftandright","upanddown","waveright"],
  "label_column": "taget",            # misspelled in data, intentional
  "n_subcarriers": 52,
  "resample": {"enabled": True, "target_fs": 100.0, "method": "linear"},
  "preprocess": {
    "hampel": {"window": 7, "n_sigmas": 3.0},
    "butter": {"cutoff_hz": 30.0, "order": 4},
    "phase": {"savgol_window": 11, "savgol_poly": 3, "linear_fit_removal": True},
    "zscore_per_subcarrier": True       # fit on TRAIN only
  },
  "window": {"length": 250, "stride": 125, "guard_windows": 1},
  "tensor": {"channels": ["amp","phase"], "shape": [2,52,250], "use_stft": False},
  "split": {"mode": "temporal_per_file", "train": 0.8, "val": 0.1, "test": 0.1},
  "encoders": ["mlp","cnn","vit"],
  "embed_dim": 128, "proj_dim": 64, "proj_hidden": 128,
  "mlp": {"hidden": 256, "dropout": 0.3, "layers": 3},
  "cnn": {"channels": [16,32,64], "dropout": 0.2},
  "vit": {"patch": [13,25], "dim": 128, "depth": 4, "heads": 4, "mlp_ratio": 2.0, "dropout": 0.1},
  "simclr": {"epochs": 200, "batch_size": 64, "drop_last": True, "temperature": 0.2,
             "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adam",
             "aug": {"noise_std": 0.1, "time_mask_frac": 0.15, "subcarrier_mask_frac": 0.10,
                     "time_shift": 10, "amp_scale": [0.9,1.1], "phase_gentler": True}},
  "linear_probe": {"epochs": 100, "batch_size": 64, "lr": 1e-3, "weight_decay": 0.0},
  "eval_mode": "linear_probe",          # or "cluster_hungarian"
  "cluster": {"k": 6, "algo": "kmeans"},
  "device": "cuda",                     # falls back to cpu
  "results_dir": "results", "plots_dir": "results/plots"
}
```

---

## Results aggregation & plots

Collect one tidy record per `(person, encoder, class)`: precision/recall/F1/support, plus run-level accuracy & macro-F1.
- **Headline table** (per encoder): mean ± std (sample std, ddof=1, n=8) of per-class P/R/F1, macro-F1, accuracy across the 8 people.
- **Raw table**: 24 person×encoder rows → `results/per_run_metrics.csv`.
- Aggregate **metrics across people** (people are the independent unit), not pooled raw predictions.
- **Plots** (`plots/visualize.py`): grouped bar chart (x=class, bars=3 encoders, y=mean F1, error=std); one confusion matrix per encoder (pooled, clearly labeled); per-person F1 box/strip plot (exposes person variance); SimCLR train/val loss curves; t-SNE of test embeddings (3 encoders side-by-side, colored by true label).

---

## Known risks (call out in the report, don't try to "fix" away)
- **Tiny per-person data**: ~234 windows/person (~39/gesture) → ~187 train after split. Borderline for SimCLR; **ViT is data-starved and will likely underperform** — report as a data limitation, not an architecture verdict.
- **Phase is noisy** on single-antenna ESP32 (no conjugate-multiply sanitization possible). Consider an **amplitude-only `[1,52,250]` ablation** — may beat 2-channel.
- BatchNorm with small batches is okay at B≈32–64; if batch drops below ~16, switch encoders to GroupNorm/LayerNorm.

---

## Setup

```bash
.conda/bin/pip install pandas scipy scikit-learn matplotlib seaborn tqdm
.conda/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only; job is tiny
# Do NOT install CSIKit.
.conda/bin/pip freeze > results/requirements.lock.txt
```
Run all commands with `.conda/bin/python`.

---

## Verification (smoke test before the 24-run sweep)

1. **Loader**: `parse_csi_csv(ID1/applause.csv)` → `iq.shape ≈ (5109,104)`, label `"applause"`, all rows len 104. `iq_to_amp_phase` → `(N,52)`, no NaN, amp ≥ 0.
2. **Resample**: uniform ~100 Hz, monotonic time, ~5109→~6000 rows.
3. **Preprocess**: no NaN/Inf; train z-scored cols mean≈0/std≈1; phase finite.
4. **Windowing**: ID1 → ~234 windows total, ~39/gesture; every tensor `== [2,52,250]`.
5. **Leakage assertion (critical)**: zero raw-time overlap between train/val/test windows; all 6 classes in each split; guard windows dropped.
6. **NT-Xent**: identical views → near-min finite loss; random views → higher; no NaN.
7. **Encoder shape contract**: random `[4,2,52,250]` → all three encoders return `[4,128]`; ViT patch grid = 4×10+CLS.
8. **One-person dry run**: CNN, `simclr.epochs=3`, `probe.epochs=3`, ID1 only → completes in <1 min on CPU, produces 6-logit probe + metrics dict for all 6 classes; test acc above chance (1/6≈0.17) **only trust after step 5 passes**.
9. **Full sweep**: 8×3 runs → tens of minutes on CPU. Hours ⇒ window/batch counts are wrong.

Then run `.conda/bin/python -m csi_pipeline.run_experiment` and inspect `results/`.
