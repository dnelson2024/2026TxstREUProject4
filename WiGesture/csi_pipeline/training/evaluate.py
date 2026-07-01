"""Prediction, metrics, and the optional clustering+Hungarian eval path."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..config import CONFIG
from .trainer import extract_embeddings


@torch.no_grad()
def predict_linear_probe(encoder, probe, X, device="cpu"):
    encoder.eval()
    probe.eval()
    Z = torch.from_numpy(extract_embeddings(encoder, X, device)).to(device)
    return probe(Z).argmax(1).cpu().numpy()


def _build_mapping(clusters, labels, k, n_classes):
    """Hungarian assignment of cluster ids -> true labels using a cost matrix."""
    cost = np.zeros((k, n_classes))
    for c, l in zip(clusters, labels):
        cost[c, l] -= 1  # negative count -> maximize overlap
    row, col = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row, col)}


def cluster_hungarian_predict(encoder, X_train, y_train, X_test, cfg=CONFIG, device="cpu"):
    """Fit k-means on TRAIN embeddings, map clusters->labels via Hungarian on
    TRAIN, then apply the frozen mapping to TEST cluster assignments."""
    k = cfg["cluster"]["k"]
    n_classes = len(cfg["classes"])
    Z_train = extract_embeddings(encoder, X_train, device)
    Z_test = extract_embeddings(encoder, X_test, device)

    km = KMeans(n_clusters=k, random_state=cfg["seed"], n_init=10)
    train_clusters = km.fit_predict(Z_train)
    mapping = _build_mapping(train_clusters, y_train, k, n_classes)

    test_clusters = km.predict(Z_test)
    return np.array([mapping.get(int(c), 0) for c in test_clusters])


def metrics(y_true, y_pred, class_names):
    """Per-class + macro precision/recall/F1, accuracy, and confusion matrix."""
    n = len(class_names)
    labels = list(range(n))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n)
        },
        "confusion_matrix": cm.tolist(),
    }
