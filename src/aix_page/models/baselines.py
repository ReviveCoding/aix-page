"""Strong prior and hashed linear baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

FEATURE_COLUMNS = ["Position", "Depth", "AdID", "AdvertiserID", "DisplayURL", "QueryID"]


def require_string_columns(frame: pd.DataFrame) -> list[str]:
    """Return validated string labels required by the feature-record contract."""
    columns: list[str] = []
    for column in frame.columns:
        if not isinstance(column, str):
            raise TypeError(
                "Hashed-linear feature frames require string column labels; "
                f"received {column!r} ({type(column).__name__})"
            )
        columns.append(column)
    return columns


def records(frame: pd.DataFrame) -> list[dict[str, float]]:
    columns = require_string_columns(frame)
    rows: list[dict[str, float]] = []
    history: list[str] = [column for column in columns if column.startswith("hist_")]
    for _, row in frame.iterrows():
        item: dict[str, float] = {f"{c}={int(row[c])}": 1.0 for c in FEATURE_COLUMNS}
        item["normalized_position"] = float(row["normalized_position"])
        item.update({c: float(row[c]) for c in history})
        rows.append(item)
    return rows


class HashedLinear:
    def __init__(self, seed: int = 0) -> None:
        self.hasher = FeatureHasher(n_features=2**16, input_type="dict", alternate_sign=True)
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=100,
            tol=1e-6,
            learning_rate="adaptive",
            eta0=0.01,
            average=True,
            random_state=seed,
        )

    def fit(self, frame: pd.DataFrame) -> HashedLinear:
        positive = frame.assign(_label=1, _weight=frame["Click"])
        negative = frame.assign(_label=0, _weight=frame["Impression"] - frame["Click"])
        stacked = pd.concat([positive, negative], ignore_index=True)
        xs = self.hasher.transform(records(stacked))
        weights = stacked["_weight"].to_numpy(dtype=np.float64)
        positive_weights = weights[weights > 0]
        if len(positive_weights) == 0:
            raise ValueError("aggregated-binomial training requires positive impression mass")
        weights /= positive_weights.mean()
        self.model.fit(xs, stacked["_label"], sample_weight=weights)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.model.predict_proba(self.hasher.transform(records(frame)))[:, 1]
        return np.asarray(probabilities, dtype=np.float64)
