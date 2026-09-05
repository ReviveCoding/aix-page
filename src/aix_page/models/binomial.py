"""Aggregated-binomial objective and weighted predictive metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -35, 35))), dtype=np.float64)


def objective(
    margin: np.ndarray, click: np.ndarray, impression: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = sigmoid(margin)
    y = click / impression
    return impression * (p - y), impression * p * (1.0 - p)


def negative_log_likelihood(margin: np.ndarray, click: np.ndarray, impression: np.ndarray) -> float:
    p = np.clip(sigmoid(margin).astype(np.float64), 1e-12, 1 - 1e-12)
    return float(-(click * np.log(p) + (impression - click) * np.log1p(-p)).sum())


def weighted_metrics(
    click: np.ndarray, impression: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    total = impression.sum()
    logloss = -(click * np.log(p) + (impression - click) * np.log1p(-p)).sum() / total
    brier = (click * (1 - p) ** 2 + (impression - click) * p**2).sum() / total
    labels = np.concatenate([np.ones(len(p)), np.zeros(len(p))])
    scores = np.concatenate([p, p])
    weights = np.concatenate([click, impression - click])
    keep = weights > 0
    auc = roc_auc_score(labels[keep], scores[keep], sample_weight=weights[keep])
    pr_auc = average_precision_score(labels[keep], scores[keep], sample_weight=weights[keep])
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    ece = sum(
        impression[b].sum()
        / total
        * abs(click[b].sum() / impression[b].sum() - np.average(p[b], weights=impression[b]))
        for b in bins
        if impression[b].sum()
    )
    return {
        "weighted_log_loss": float(logloss),
        "weighted_brier": float(brier),
        "weighted_roc_auc": float(auc),
        "weighted_pr_auc": float(pr_auc),
        "ece": float(ece),
    }
