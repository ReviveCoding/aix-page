"""Calibration fitted exclusively on the calibration partition."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def pseudo_rows(
    click: np.ndarray, impression: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([score, score])[:, None]
    y = np.concatenate([np.ones(len(score)), np.zeros(len(score))])
    w = np.concatenate([click, impression - click])
    keep = w > 0
    return x[keep], y[keep], w[keep]


class Platt:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6)

    def fit(self, click: np.ndarray, impression: np.ndarray, p: np.ndarray) -> Platt:
        x, y, w = pseudo_rows(
            click, impression, np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))
        )
        self.model.fit(x, y, sample_weight=w)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        logit = np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))
        return np.asarray(self.model.predict_proba(logit[:, None])[:, 1], dtype=np.float64)


class Isotonic:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, click: np.ndarray, impression: np.ndarray, p: np.ndarray) -> Isotonic:
        x, y, w = pseudo_rows(click, impression, p)
        self.model.fit(x[:, 0], y, sample_weight=w)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(p))
