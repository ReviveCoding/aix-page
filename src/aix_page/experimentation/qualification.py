"""A/A and power qualification."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def run_aa(
    replications: int, seed: int, n_per_arm: int = 1200
) -> dict[str, float | int | bool | list[float]]:
    rng = np.random.default_rng(seed)
    rejections = 0
    covers = 0
    for _ in range(replications):
        a = rng.normal(0.1, 0.3, n_per_arm)
        b = rng.normal(0.1, 0.3, n_per_arm)
        effect = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / n_per_arm + b.var(ddof=1) / n_per_arm)
        rejections += abs(effect / se) > norm.ppf(0.975)
        covers += effect - 1.96 * se <= 0 <= effect + 1.96 * se
    type_i = rejections / replications
    coverage = covers / replications
    type_i_mc_se = float(np.sqrt(type_i * (1 - type_i) / replications))
    coverage_mc_se = float(np.sqrt(coverage * (1 - coverage) / replications))
    return {
        "replications": replications,
        "type_i_error": type_i,
        "coverage_95": coverage,
        "type_i_mc_ci": [
            max(0.0, type_i - 1.96 * type_i_mc_se),
            min(1.0, type_i + 1.96 * type_i_mc_se),
        ],
        "coverage_mc_ci": [
            max(0.0, coverage - 1.96 * coverage_mc_se),
            min(1.0, coverage + 1.96 * coverage_mc_se),
        ],
        "pass": bool(type_i <= 0.09 and coverage >= 0.91),
    }


def power_analysis(
    sd: float, effect: float, alpha: float = 0.05, target: float = 0.8
) -> dict[str, float | int]:
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(target)
    per_arm = int(np.ceil(2 * (z_alpha + z_power) ** 2 * sd**2 / effect**2))
    achieved = float(norm.cdf(abs(effect) * np.sqrt(per_arm / (2 * sd**2)) - z_alpha))
    return {
        "sd": sd,
        "minimum_detectable_effect": effect,
        "target_power": target,
        "required_per_arm": per_arm,
        "required_total_12_arms": per_arm * 12,
        "achieved_power": achieved,
    }
