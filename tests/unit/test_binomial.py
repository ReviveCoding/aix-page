import numpy as np

from aix_page.models.binomial import negative_log_likelihood, objective, sigmoid, weighted_metrics


def test_gradient_and_hessian_finite_difference() -> None:
    z = np.array([-0.8, 0.4])
    click = np.array([2.0, 7.0])
    impression = np.array([10.0, 12.0])
    gradient, hessian = objective(z, click, impression)
    eps = 1e-5
    for i in range(2):
        direction = np.zeros(2)
        direction[i] = eps
        numeric_gradient = (
            negative_log_likelihood(z + direction, click, impression)
            - negative_log_likelihood(z - direction, click, impression)
        ) / (2 * eps)
        plus_gradient = objective(z + direction, click, impression)[0][i]
        minus_gradient = objective(z - direction, click, impression)[0][i]
        numeric_hessian = (plus_gradient - minus_gradient) / (2 * eps)
        assert np.isclose(gradient[i], numeric_gradient, rtol=1e-5)
        assert np.isclose(hessian[i], numeric_hessian, rtol=1e-5)


def test_grouped_likelihood_equals_expanded() -> None:
    z = np.array([0.2, -0.6])
    click = np.array([2, 1])
    impression = np.array([3, 4])
    grouped = negative_log_likelihood(z, click, impression)
    expanded_z = np.repeat(z, impression)
    expanded_y = np.concatenate(
        [np.r_[np.ones(c), np.zeros(n - c)] for c, n in zip(click, impression, strict=True)]
    )
    p = sigmoid(expanded_z)
    expanded = -(expanded_y * np.log(p) + (1 - expanded_y) * np.log1p(-p)).sum()
    assert np.isclose(grouped, expanded)


def test_grouped_auc_equals_expanded() -> None:
    click = np.array([2, 0, 3])
    impression = np.array([4, 2, 3])
    p = np.array([0.3, 0.1, 0.8])
    grouped = weighted_metrics(click, impression, p)["weighted_roc_auc"]
    y = np.concatenate(
        [np.r_[np.ones(c), np.zeros(n - c)] for c, n in zip(click, impression, strict=True)]
    )
    score = np.repeat(p, impression)
    from sklearn.metrics import roc_auc_score

    assert np.isclose(grouped, roc_auc_score(y, score))


def test_optimization_direction_reduces_loss() -> None:
    z = np.array([0.0])
    click = np.array([8.0])
    impression = np.array([10.0])
    gradient, hessian = objective(z, click, impression)
    assert negative_log_likelihood(
        z - 0.2 * gradient / hessian, click, impression
    ) < negative_log_likelihood(z, click, impression)


def test_float32_extreme_probabilities_produce_finite_metrics() -> None:
    metrics = weighted_metrics(
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([0.0, 1.0], dtype=np.float32),
    )
    assert all(np.isfinite(value) for value in metrics.values())
