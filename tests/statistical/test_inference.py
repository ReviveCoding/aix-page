import numpy as np
import pandas as pd

from aix_page.constants import CONTROL, TREATMENTS
from aix_page.experimentation.inference import (
    bh_fdr,
    cluster_bootstrap_difference,
    cuped,
    factorial_jackknife,
    holm,
    jackknife_difference,
    noninferiority,
    regression_adjusted_difference,
    srm_test,
)
from aix_page.experimentation.qualification import run_aa


def test_srm_known_examples() -> None:
    balanced = pd.DataFrame({"treatment": np.repeat(TREATMENTS, 100)})
    assert srm_test(balanced)["pass"] is True
    skewed = pd.DataFrame({"treatment": [CONTROL] * 1100 + list(TREATMENTS[1:]) * 10})
    assert srm_test(skewed)["pass"] is False


def test_srm_uses_randomization_unit_for_clustered_assignment() -> None:
    rng = np.random.default_rng(81)
    users = pd.DataFrame(
        {
            "sim_user_id": [f"u{i}" for i in range(12000)],
            "treatment": np.repeat(TREATMENTS, 1000),
            "sessions": rng.integers(1, 7, 12000),
        }
    )
    sessions = users.loc[users.index.repeat(users.sessions), ["sim_user_id", "treatment"]]
    randomized_units = sessions.drop_duplicates("sim_user_id")
    assert srm_test(randomized_units)["pass"] is True


def test_jackknife_known_constant_effect() -> None:
    rows = []
    for bucket in range(20):
        rows += [
            {"treatment": CONTROL, "jackknife_bucket": bucket, "y": 1.0},
            {"treatment": TREATMENTS[1], "jackknife_bucket": bucket, "y": 1.5},
        ]
    result = jackknife_difference(pd.DataFrame(rows), "y", TREATMENTS[1])
    assert result["effect"] == 0.5 and result["se"] == 0.0


def test_cuped_null_unbiased_and_reduces_variance() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=10000)
    treatment = rng.binomial(1, 0.5, 10000)
    y = 2 * x + rng.normal(size=10000)
    adjusted, _ = cuped(pd.DataFrame({"x": x, "y": y}), "y", "x")
    assert abs(adjusted[treatment == 1].mean() - adjusted[treatment == 0].mean()) < 0.05
    assert adjusted.var() < y.var()


def test_multiple_and_noninferiority_rules() -> None:
    adjusted = holm([0.01, 0.04, 0.2])
    assert all(a >= p for a, p in zip(adjusted, [0.01, 0.04, 0.2], strict=True))
    assert noninferiority(-0.002, 0.002, -0.015, True)
    assert noninferiority(0.002, 0.002, 0.01, False)


def test_bh_known_answer_and_real_sensitivity_estimators() -> None:
    assert np.allclose(bh_fdr([0.01, 0.04, 0.03, 0.20]), [0.04, 0.0533333333, 0.0533333333, 0.20])
    rng = np.random.default_rng(44)
    n = 4000
    assigned = rng.binomial(1, 0.5, n)
    x = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "treatment": np.where(assigned == 1, TREATMENTS[1], CONTROL),
            "sim_user_id": [f"u{i}" for i in range(n)],
            "query_cluster": rng.integers(0, 100, n),
            "x": x,
            "y": 0.2 * assigned + x + rng.normal(size=n),
        }
    )
    adjusted = regression_adjusted_difference(frame, "y", TREATMENTS[1], ["x"])
    boot = cluster_bootstrap_difference(
        frame, "y", TREATMENTS[1], "query_cluster", replications=80, seed=5
    )
    assert abs(adjusted["effect"] - 0.2) < 0.08
    assert boot["se"] > 0 and boot["replications"] == 80


def test_aa_null_behavior() -> None:
    result = run_aa(120, 9, 500)
    assert result["pass"]


def test_factorial_model_recovers_known_cell_effects() -> None:
    rows = []
    for bucket in range(20):
        for treatment in TREATMENTS:
            fmt, position, density = treatment.split("__")
            mean = 1.0 + 0.2 * (fmt == "rich_product") + 0.1 * (position == "inline")
            mean += 0.05 * (density == "two") + 0.03 * (
                fmt == "rich_product" and position == "inline"
            )
            rows.append(
                {
                    "treatment": treatment,
                    "jackknife_bucket": bucket,
                    "metric_sum": mean * 100,
                    "metric_count": 100,
                }
            )
    result = factorial_jackknife(pd.DataFrame(rows)).set_index("term")
    assert np.isclose(result.loc["format_rich", "effect"], 0.2)
    assert np.isclose(result.loc["rich_x_inline", "effect"], 0.03)
