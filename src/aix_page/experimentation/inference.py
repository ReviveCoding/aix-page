"""Trust checks, jackknife, CUPED, multiplicity and non-inferiority."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests

from aix_page.constants import CONTROL, TREATMENTS


def _factor_design(treatment: str) -> list[float]:
    fmt, position, density = treatment.split("__")
    rich = float(fmt == "rich_product")
    conversational = float(fmt == "conversational_sponsored")
    inline = float(position == "inline")
    two = float(density == "two")
    return [
        1.0,
        rich,
        conversational,
        inline,
        two,
        rich * inline,
        conversational * inline,
        rich * two,
        conversational * two,
    ]


def factorial_jackknife(bucket_aggregates: pd.DataFrame) -> pd.DataFrame:
    """Factorial WLS coefficients with leave-one-user-bucket jackknife SEs."""
    names = [
        "intercept",
        "format_rich",
        "format_conversational",
        "position_inline",
        "density_two",
        "rich_x_inline",
        "conversational_x_inline",
        "rich_x_two",
        "conversational_x_two",
    ]

    def fit(block: pd.DataFrame) -> np.ndarray:
        cells = block.groupby("treatment", as_index=False)[["metric_sum", "metric_count"]].sum()
        x = np.asarray([_factor_design(str(t)) for t in cells["treatment"]], dtype=float)
        y = cells["metric_sum"].to_numpy(float) / cells["metric_count"].to_numpy(float)
        root_w = np.sqrt(cells["metric_count"].to_numpy(float))
        coefficients = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0]
        return np.asarray(coefficients, dtype=float)

    estimate = fit(bucket_aggregates)
    buckets = sorted(bucket_aggregates["jackknife_bucket"].unique())
    leave = np.asarray(
        [
            fit(bucket_aggregates[bucket_aggregates.jackknife_bucket.ne(bucket)])
            for bucket in buckets
        ]
    )
    se = np.sqrt(
        (len(buckets) - 1) / len(buckets) * np.square(leave - leave.mean(axis=0)).sum(axis=0)
    )
    z = np.divide(estimate, se, out=np.zeros_like(estimate), where=se > 0)
    return pd.DataFrame(
        {
            "term": names,
            "effect": estimate,
            "se": se,
            "ci_low": estimate - 1.96 * se,
            "ci_high": estimate + 1.96 * se,
            "p_value": 2 * norm.sf(np.abs(z)),
        }
    )


def srm_test(frame: pd.DataFrame) -> dict[str, float | bool]:
    counts = frame.groupby("treatment").size().reindex(TREATMENTS, fill_value=0).to_numpy()
    expected = counts.sum() / len(TREATMENTS)
    statistic = float(((counts - expected) ** 2 / expected).sum())
    p = float(chi2.sf(statistic, len(TREATMENTS) - 1))
    return {"statistic": statistic, "p_value": p, "pass": p >= 0.001}


def jackknife_difference(
    frame: pd.DataFrame, metric: str, treatment: str, control: str = CONTROL
) -> dict[str, float]:
    subset = frame[frame["treatment"].isin([treatment, control])]
    effect = (
        subset.loc[subset.treatment.eq(treatment), metric].mean()
        - subset.loc[subset.treatment.eq(control), metric].mean()
    )
    estimates = []
    for bucket in range(20):
        leave = subset[subset["jackknife_bucket"] != bucket]
        estimates.append(
            leave.loc[leave.treatment.eq(treatment), metric].mean()
            - leave.loc[leave.treatment.eq(control), metric].mean()
        )
    estimates_array = np.asarray(estimates)
    se = float(np.sqrt(19 / 20 * np.square(estimates_array - estimates_array.mean()).sum()))
    z = effect / se if se > 0 else 0.0
    return {
        "effect": float(effect),
        "se": se,
        "ci_low": float(effect - 1.96 * se),
        "ci_high": float(effect + 1.96 * se),
        "p_value": float(2 * norm.sf(abs(z))),
    }


def cuped(frame: pd.DataFrame, metric: str, covariate: str) -> tuple[np.ndarray, float]:
    x = frame[covariate].to_numpy(float)
    y = frame[metric].to_numpy(float)
    variance = np.var(x)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / variance) if variance > 0 else 0.0
    return y - theta * (x - x.mean()), theta


def holm(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values in the original order."""
    if not p_values:
        return []
    adjusted = np.asarray(multipletests(p_values, method="fdr_bh")[1], dtype=float)
    return [float(value) for value in adjusted]


def regression_adjusted_difference(
    frame: pd.DataFrame,
    metric: str,
    treatment: str,
    covariates: list[str],
    control: str = CONTROL,
) -> dict[str, float]:
    """ANCOVA treatment coefficient with user-cluster robust uncertainty."""
    import statsmodels.api as sm

    subset = frame[frame["treatment"].isin([control, treatment])].copy()
    indicator = subset["treatment"].eq(treatment).astype(float).rename("treatment_indicator")
    design = sm.add_constant(pd.concat([indicator, subset[covariates]], axis=1), has_constant="add")
    fitted = sm.OLS(subset[metric].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": subset["sim_user_id"]}
    )
    effect = float(fitted.params["treatment_indicator"])
    se = float(fitted.bse["treatment_indicator"])
    return {
        "effect": effect,
        "se": se,
        "ci_low": effect - 1.96 * se,
        "ci_high": effect + 1.96 * se,
        "p_value": float(fitted.pvalues["treatment_indicator"]),
    }


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    metric: str,
    treatment: str,
    cluster: str,
    *,
    replications: int,
    seed: int,
    control: str = CONTROL,
) -> dict[str, float]:
    """Cluster bootstrap for a treatment-minus-control mean difference."""
    subset = frame[frame["treatment"].isin([control, treatment])]
    clusters = subset[cluster].drop_duplicates().to_numpy()
    grouped = subset.groupby([cluster, "treatment"])[metric].agg(["sum", "count"]).reset_index()
    clusters_index = pd.Index(clusters)
    sums = (
        grouped.pivot(index=cluster, columns="treatment", values="sum")
        .reindex(index=clusters_index, columns=[control, treatment], fill_value=0)
        .fillna(0)
        .to_numpy(float)
    )
    counts = (
        grouped.pivot(index=cluster, columns="treatment", values="count")
        .reindex(index=clusters_index, columns=[control, treatment], fill_value=0)
        .fillna(0)
        .to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replications):
        sampled = rng.integers(0, len(clusters), len(clusters))
        weights = np.bincount(sampled, minlength=len(clusters)).astype(float)
        arm_sum = weights @ sums
        arm_count = weights @ counts
        if np.all(arm_count > 0):
            estimates.append(float(arm_sum[1] / arm_count[1] - arm_sum[0] / arm_count[0]))
    values = np.asarray(estimates)
    effect = float(
        subset.loc[subset.treatment.eq(treatment), metric].mean()
        - subset.loc[subset.treatment.eq(control), metric].mean()
    )
    return {
        "effect": effect,
        "se": float(values.std(ddof=1)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "replications": float(len(values)),
    }


def noninferiority(effect: float, se: float, margin: float, higher_is_better: bool = True) -> bool:
    boundary = effect - 1.645 * se if higher_is_better else effect + 1.645 * se
    return bool(boundary > margin if higher_is_better else boundary < margin)
