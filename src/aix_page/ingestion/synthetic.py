"""Contract-shaped synthetic fixture; never represented as KDD evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_aggregated_fixture(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    query = rng.integers(1, max(100, n // 30), n)
    ad = rng.integers(1, max(80, n // 50), n)
    advertiser = ad // 7 + 1
    user = rng.integers(0, max(500, n // 8), n)
    user[rng.random(n) < 0.12] = 0
    depth = rng.integers(1, 5, n)
    position = np.array([rng.integers(1, d + 1) for d in depth])
    relevance = rng.beta(3, 2, n)
    logit = -3.4 - 0.42 * (position - 1) + 0.8 * relevance + 0.15 * (advertiser % 7 == 0)
    probability = 1 / (1 + np.exp(-logit))
    impression = rng.integers(1, 31, n)
    click = rng.binomial(impression, probability)
    return pd.DataFrame(
        {
            "Click": click,
            "Impression": impression,
            "DisplayURL": ad % 311,
            "AdID": ad,
            "AdvertiserID": advertiser,
            "Depth": depth,
            "Position": position,
            "QueryID": query,
            "KeywordID": query % 1901,
            "TitleID": ad % 1301,
            "DescriptionID": ad % 1709,
            "UserID": user,
            "synthetic_relevance": relevance,
        }
    )
