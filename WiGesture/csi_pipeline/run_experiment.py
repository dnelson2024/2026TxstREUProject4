"""End-to-end orchestration: 8 people x 3 encoders.

For each person: load -> resample -> clean -> window -> temporal split (leakage
safe) -> fit z-score on train. For each encoder: SimCLR pretrain -> freeze ->
linear probe (or cluster+Hungarian) -> metrics. Results aggregated and plotted.

Usage:
    python -m csi_pipeline.run_experiment            # full sweep
    python -m csi_pipeline.run_experiment --smoke    # 1 person, CNN, 3 epochs
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG, resolve_device
from .data import preprocess as pp
from .data import windowing as wd
from .data.loader import load_person_recordings
from .models import build_encoder
from .training import evaluate as ev
from .training import trainer as tr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PersonData:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray


def prepare_person(person_id: str, cfg=CONFIG) -> PersonData | None:
    """Load, clean, window, split, and tensorize one person's data (shared
    across all encoders so it's computed once)."""
    recs = load_person_recordings(person_id, cfg["data_root"])
    if not recs:
        print(f"  [{person_id}] no recordings, skipping")
        return None

    # Clean each full recording before windowing.
    for rec in recs:
        rec.amp = pp.amplitude_pipeline(rec.amp, rec.fs)
        rec.phase = pp.phase_pipeline(rec.phase)

    train, val, test = wd.split_person(recs, cfg)
    if not train or not test:
        print(f"  [{person_id}] empty split, skipping")
        return None

    stats = wd.fit_zscore_stats(train) if cfg["preprocess"]["zscore_per_subcarrier"] else None
    classes = cfg["classes"]
    Xtr, ytr = wd.windows_to_arrays(train, stats, classes)
    Xva, yva = wd.windows_to_arrays(val, stats, classes) if val else (np.empty((0, *cfg["tensor"]["shape"]), np.float32), np.empty((0,), np.int64))
    Xte, yte = wd.windows_to_arrays(test, stats, classes)
    print(f"  [{person_id}] windows: train {len(ytr)}  val {len(yva)}  test {len(yte)}")
    return PersonData(Xtr, ytr, Xva, yva, Xte, yte)


def run_one(person_id, encoder_name, data: PersonData, cfg, device, save_history=None):
    set_seed(cfg["seed"])
    encoder = build_encoder(encoder_name, cfg)
    encoder, history = pretrain_and_history(encoder, data, cfg, device)
    if save_history is not None:
        save_history[(person_id, encoder_name)] = history
    tr.freeze(encoder)

    n_classes = len(cfg["classes"])
    if cfg["eval_mode"] == "cluster_hungarian":
        y_pred = ev.cluster_hungarian_predict(encoder, data.Xtr, data.ytr, data.Xte, cfg, device)
    else:
        probe = tr.train_linear_probe(
            encoder, data.Xtr, data.ytr, data.Xva, data.yva, n_classes, cfg, device
        )
        y_pred = ev.predict_linear_probe(encoder, probe, data.Xte, device)
    m = ev.metrics(data.yte, y_pred, cfg["classes"])
    print(f"  [{person_id}/{encoder_name}] test acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}")
    return m, encoder


def pretrain_and_history(encoder, data, cfg, device):
    val_X = data.Xva if len(data.Xva) >= cfg["simclr"]["batch_size"] else None
    return tr.pretrain_contrastive(encoder, data.Xtr, cfg, device, val_X=val_X, verbose=True)


def aggregate(records, cfg):
    """Build tidy per-class dataframe and per-(encoder,class) F1 mean/std."""
    rows = []
    for rec in records:
        for cname, vals in rec["per_class"].items():
            rows.append({
                "person": rec["person"],
                "encoder": rec["encoder"],
                "class": cname,
                "precision": vals["precision"],
                "recall": vals["recall"],
                "f1": vals["f1"],
                "support": vals["support"],
            })
    df = pd.DataFrame(rows)
    agg = {}
    for enc in cfg["encoders"]:
        for c in cfg["classes"]:
            sub = df[(df.encoder == enc) & (df["class"] == c)]["f1"]
            agg[(enc, c)] = (float(sub.mean()) if len(sub) else 0.0,
                             float(sub.std(ddof=1)) if len(sub) > 1 else 0.0)
    return df, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 person, CNN, 3 epochs")
    args = ap.parse_args()

    cfg = json.loads(json.dumps(CONFIG))  # deep copy so smoke edits don't persist
    people = cfg["person_ids"]
    encoders = cfg["encoders"]
    if args.smoke:
        people = people[:1]
        encoders = ["cnn"]
        cfg["simclr"]["epochs"] = 3
        cfg["linear_probe"]["epochs"] = 3

    set_seed(cfg["seed"])
    device = resolve_device(cfg)
    print(f"Device: {device} | people: {people} | encoders: {encoders} | eval: {cfg['eval_mode']}")

    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(cfg["plots_dir"], exist_ok=True)

    records = []          # one dict per (person, encoder)
    histories = {}
    pooled_cm = {e: np.zeros((len(cfg["classes"]),) * 2, dtype=float) for e in encoders}

    for pid in people:
        data = prepare_person(pid, cfg)
        if data is None:
            continue
        for enc in encoders:
            m, _ = run_one(pid, enc, data, cfg, device, save_history=histories)
            rec = {"person": pid, "encoder": enc, **{k: m[k] for k in
                   ("accuracy", "macro_precision", "macro_recall", "macro_f1")},
                   "per_class": m["per_class"]}
            records.append(rec)
            pooled_cm[enc] += np.array(m["confusion_matrix"], dtype=float)

    # --- write results ---
    df_runs = pd.DataFrame([{k: v for k, v in r.items() if k != "per_class"} for r in records])
    df_runs.to_csv(os.path.join(cfg["results_dir"], "per_run_metrics.csv"), index=False)

    df_class, agg = aggregate(records, cfg)
    df_class.to_csv(os.path.join(cfg["results_dir"], "per_class_metrics.csv"), index=False)

    summary = (df_runs.groupby("encoder")[["accuracy", "macro_f1"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(os.path.join(cfg["results_dir"], "summary.csv"))
    print("\n=== Summary (mean/std over people) ===")
    print(summary)

    # --- plots ---
    try:
        from .plots import visualize as vz

        for enc in encoders:
            vz.plot_confusion(pooled_cm[enc], cfg["classes"],
                              f"Pooled confusion: {enc}",
                              os.path.join(cfg["plots_dir"], f"confusion_{enc}.png"))
        vz.plot_f1_bars(agg, cfg["classes"], encoders,
                        os.path.join(cfg["plots_dir"], "f1_by_encoder.png"))
        vz.plot_per_person_f1(records, encoders,
                              os.path.join(cfg["plots_dir"], "per_person_f1.png"))
        # one representative training curve per encoder (first person available)
        for enc in encoders:
            for (pid, e), h in histories.items():
                if e == enc:
                    vz.plot_training_curves(h, f"SimCLR loss: {pid}/{enc}",
                                            os.path.join(cfg["plots_dir"], f"loss_{enc}.png"))
                    break
        print(f"Plots written to {cfg['plots_dir']}")
    except Exception as e:  # plots are non-essential
        print(f"[warn] plotting failed: {e}")


if __name__ == "__main__":
    main()
