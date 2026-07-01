"""Single source of truth for pipeline configuration.

Everything tunable lives in CONFIG. Modules import this dict and read from it;
they never hard-code hyperparameters.
"""

CONFIG = {
    "seed": 42,
    "data_root": "dataset/dynamic",
    "person_ids": ["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8"],
    "classes": [
        "applause",
        "circleclockwise",
        "frontandafter",
        "leftandright",
        "upanddown",
        "waveright",
    ],
    "label_column": "taget",  # misspelled in the dataset, intentional
    "n_subcarriers": 52,

    "resample": {"enabled": True, "target_fs": 100.0, "method": "linear"},

    "preprocess": {
        "hampel": {"window": 7, "n_sigmas": 3.0},
        "butter": {"cutoff_hz": 30.0, "order": 4},
        "phase": {"savgol_window": 11, "savgol_poly": 3, "linear_fit_removal": True},
        "zscore_per_subcarrier": True,  # fit on TRAIN only
    },

    "window": {"length": 250, "stride": 125, "guard_windows": 1},

    "tensor": {"channels": ["amp", "phase"], "shape": [2, 52, 250], "use_stft": False},

    "split": {"mode": "temporal_per_file", "train": 0.8, "val": 0.1, "test": 0.1},

    "encoders": ["mlp", "cnn", "vit"],
    "embed_dim": 128,
    "proj_dim": 64,
    "proj_hidden": 128,

    "mlp": {"hidden": 256, "dropout": 0.3, "layers": 3},
    "cnn": {"channels": [16, 32, 64], "dropout": 0.2},
    "vit": {
        "patch": [13, 25],  # 52/13=4, 250/25=10 -> 40 patches
        "dim": 128,
        "depth": 4,
        "heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.1,
    },

    "simclr": {
        "epochs": 200,
        "batch_size": 64,
        "drop_last": True,
        "temperature": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "aug": {
            "noise_std": 0.1,
            "time_mask_frac": 0.15,
            "subcarrier_mask_frac": 0.10,
            "time_shift": 10,
            "amp_scale": [0.9, 1.1],
            "phase_gentler": True,
        },
    },

    "linear_probe": {"epochs": 100, "batch_size": 64, "lr": 1e-3, "weight_decay": 0.0},

    "eval_mode": "linear_probe",  # or "cluster_hungarian"
    "cluster": {"k": 6, "algo": "kmeans"},

    "device": "cuda",  # falls back to cpu if unavailable
    "results_dir": "results",
    "plots_dir": "results/plots",
}


def resolve_device(cfg=CONFIG):
    """Return the configured device, falling back to CPU when CUDA is absent."""
    import torch

    if cfg["device"] == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
