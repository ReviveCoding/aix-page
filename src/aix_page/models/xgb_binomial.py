"""XGBoost aggregated-binomial objective with exact-once impression weighting."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from aix_page.models.binomial import sigmoid


def aggregated_binomial_objective(
    margin: np.ndarray, matrix: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    probability = sigmoid(margin)
    fraction = matrix.get_label()
    impression = matrix.get_weight()
    if len(impression) == 0:
        impression = np.ones_like(fraction)
    gradient = impression * (probability - fraction)
    hessian = impression * probability * (1.0 - probability)
    return gradient, hessian
