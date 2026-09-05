import numpy as np
import pandas as pd

from aix_page.models.baselines import HashedLinear
from aix_page.models.binomial import weighted_metrics


def test_hashed_linear_materially_beats_prior_on_known_linear_signal() -> None:
    rng = np.random.default_rng(104)
    rows = 30_000
    position = rng.integers(1, 5, rows)
    depth = np.maximum(position, rng.integers(1, 5, rows))
    relevance = rng.normal(size=rows)
    margin = -3.0 - 0.55 * (position - 1) + 1.25 * relevance
    probability = 1 / (1 + np.exp(-margin))
    impression = rng.integers(1, 25, rows)
    click = rng.binomial(impression, probability)
    frame = pd.DataFrame(
        {
            "Click": click,
            "Impression": impression,
            "Position": position,
            "Depth": depth,
            "AdID": rng.integers(1, 40, rows),
            "AdvertiserID": rng.integers(1, 10, rows),
            "DisplayURL": rng.integers(1, 20, rows),
            "QueryID": rng.integers(1, 100, rows),
            "normalized_position": position / depth,
            "hist_known_signal": 1 / (1 + np.exp(-(-3.0 + 1.25 * relevance))),
        }
    )
    train = frame.iloc[:24_000]
    test = frame.iloc[24_000:]
    prior = train["Click"].sum() / train["Impression"].sum()
    candidate = HashedLinear(seed=104).fit(train).predict(test)
    baseline_loss = weighted_metrics(
        test["Click"].to_numpy(),
        test["Impression"].to_numpy(),
        np.full(len(test), prior),
    )["weighted_log_loss"]
    candidate_loss = weighted_metrics(
        test["Click"].to_numpy(), test["Impression"].to_numpy(), candidate
    )["weighted_log_loss"]
    assert candidate_loss < baseline_loss - 0.02
