"""Plotting helpers. All functions save PNGs and return the output path."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _ensure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def plot_confusion(cm, labels, title, path):
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(7, 6))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=120)
    plt.close()
    return path


def plot_training_curves(history, title, path):
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="train")
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("NT-Xent loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=120)
    plt.close()
    return path


def plot_embedding_tsne(embeddings_by_encoder, labels, class_names, path):
    """embeddings_by_encoder: dict[name -> (Z[N,2] 2D tsne, y[N])]."""
    n = len(embeddings_by_encoder)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (name, (Z2, y)) in zip(axes[0], embeddings_by_encoder.items()):
        for ci, cname in enumerate(class_names):
            m = y == ci
            ax.scatter(Z2[m, 0], Z2[m, 1], s=10, label=cname, alpha=0.6)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=120)
    plt.close()
    return path


def plot_f1_bars(agg, class_names, encoders, path):
    """agg: dict[(encoder,class) -> (mean,std)] of per-class F1."""
    x = np.arange(len(class_names))
    width = 0.8 / len(encoders)
    plt.figure(figsize=(11, 6))
    for i, enc in enumerate(encoders):
        means = [agg[(enc, c)][0] for c in class_names]
        stds = [agg[(enc, c)][1] for c in class_names]
        plt.bar(x + i * width, means, width, yerr=stds, capsize=3, label=enc)
    plt.xticks(x + width * (len(encoders) - 1) / 2, class_names, rotation=45, ha="right")
    plt.ylabel("F1 (mean +/- std over 8 people)")
    plt.title("Per-class F1 by encoder")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=120)
    plt.close()
    return path


def plot_per_person_f1(records, encoders, path):
    """records: list of dicts with keys 'encoder','macro_f1'. Box plot per encoder."""
    data = [[r["macro_f1"] for r in records if r["encoder"] == e] for e in encoders]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=encoders, showmeans=True)
    plt.ylabel("macro-F1 per person")
    plt.title("Per-person macro-F1 spread by encoder")
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=120)
    plt.close()
    return path
