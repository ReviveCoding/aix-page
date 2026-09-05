"""Stable group partitions and leakage-safe response history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aix_page.utils.hashing import stable_mod

PARTITIONS = (
    "model_train",
    "model_validation",
    "model_calibration",
    "model_locked_test",
    "experiment_context_pool",
    "challenge_pool",
)


def setting_key(row: pd.Series) -> str:
    return "|".join(str(int(row[c])) for c in ["UserID", "AdID", "QueryID", "Position", "Depth"])


def assign_partitions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    keys = out.apply(setting_key, axis=1)
    bucket = keys.map(lambda value: stable_mod(100, "model_partition_v1", value))
    labels = np.select(
        [bucket < 50, bucket < 60, bucket < 70, bucket < 80, bucket < 95],
        PARTITIONS[:5],
        default=PARTITIONS[5],
    )
    out["setting_key"] = keys
    out["partition"] = labels
    return out


def _priors(history: pd.DataFrame, key: str, alpha: float = 20.0) -> dict[int, float]:
    global_ctr = history["Click"].sum() / history["Impression"].sum()
    grouped = history.groupby(key, observed=True)[["Click", "Impression"]].sum()
    estimates = (grouped["Click"] + alpha * global_ctr) / (grouped["Impression"] + alpha)
    priors: dict[int, float] = {}
    for identifier, estimate in estimates.items():
        if not isinstance(identifier, (int, np.integer)):
            raise TypeError(f"CTR-prior identifiers must be integers; received {identifier!r}")
        priors[int(identifier)] = float(estimate)
    return priors


def crossfit_features(frame: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    out = frame.copy()
    train_mask = out["partition"].eq("model_train")
    train = out.loc[train_mask]
    global_ctr = train["Click"].sum() / train["Impression"].sum()
    out["normalized_position"] = out["Position"] / out["Depth"]
    for key in ["AdID", "AdvertiserID", "DisplayURL", "QueryID"]:
        column = f"hist_{key}_ctr"
        out[column] = np.nan
        fold_id = out["setting_key"].map(lambda x: stable_mod(folds, "oof_v1", x))
        for fold in range(folds):
            target = train_mask & fold_id.eq(fold)
            history = out.loc[train_mask & ~fold_id.eq(fold)]
            mapping = _priors(history, key)
            out.loc[target, column] = out.loc[target, key].map(mapping).fillna(global_ctr)
        mapping = _priors(train, key)
        out.loc[~train_mask, column] = out.loc[~train_mask, key].map(mapping).fillna(global_ctr)
    out["cold_query"] = ~out["QueryID"].isin(set(train["QueryID"]))
    out["cold_ad"] = ~out["AdID"].isin(set(train["AdID"]))
    out["anonymous_user"] = out["UserID"].eq(0)
    return out
