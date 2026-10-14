"""Confirm Platt log-odds clipping remains numerically identical."""

import numpy as np

from aix_page.calibration.calibrators import _logit, pseudo_rows


def test_logit_reuses_exact_historical_clipping_formula():
    probabilities = np.array([0.0, 1e-9, 0.1, 0.5, 0.9, 1.0])
    expected = np.log(
        np.clip(probabilities, 1e-8, 1 - 1e-8)
        / np.clip(1 - probabilities, 1e-8, 1)
    )
    np.testing.assert_array_equal(_logit(probabilities), expected)


def test_pseudo_rows_keep_positive_and_negative_weights():
    x, labels, weights = pseudo_rows(
        np.array([2.0, 0.0]),
        np.array([10.0, 5.0]),
        np.array([0.2, 0.8]),
    )
    assert x.shape == (3, 1)
    assert labels.tolist() == [1.0, 0.0, 0.0]
    assert weights.tolist() == [2.0, 8.0, 5.0]
