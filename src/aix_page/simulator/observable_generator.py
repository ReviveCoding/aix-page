"""Observable semi-synthetic outcomes. Contains no oracle evaluation API."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aix_page.experimentation.assignment import jackknife_bucket, policy_partition, treatment_for


def generate_context(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user = rng.integers(1, max(2000, n // 3), n).astype(str)
    return pd.DataFrame(
        {
            "sim_user_id": [f"u{x}" for x in user],
            "query_cluster": rng.integers(0, max(200, n // 40), n),
            "commercial_intent": rng.binomial(1, 0.48, n),
            "relevance": rng.beta(4, 2, n),
            "tail_query": rng.binomial(1, 0.28, n),
            "returning_user": rng.binomial(1, 0.64, n),
            "prior_exposure": rng.poisson(2.2, n),
            "historical_pctr": rng.beta(2, 40, n),
            "prior_engagement": rng.beta(4, 3, n),
            "prior_advertiser_value": rng.gamma(1.8, 0.12, n),
        }
    )


def potential_means(context: pd.DataFrame, treatment: str) -> dict[str, np.ndarray]:
    fmt, pos, density = treatment.split("__")
    commercial = context["commercial_intent"].to_numpy()
    relevance = context["relevance"].to_numpy()
    tail = context["tail_query"].to_numpy()
    exposure = context["prior_exposure"].to_numpy()
    rich = float(fmt == "rich_product")
    conversation = float(fmt == "conversational_sponsored")
    inline = float(pos == "inline")
    two = float(density == "two")
    top = 1.0 - inline
    base_click = np.clip(context["historical_pctr"].to_numpy() * 1.4 + 0.01, 0.01, 0.35)
    click = np.clip(
        base_click
        + 0.018 * rich * commercial
        + 0.014 * conversation * tail
        + 0.008 * top
        - 0.008 * two * (1 - commercial)
        - 0.006 * exposure / 6,
        0.002,
        0.65,
    )
    conversion = np.clip(0.06 + 0.04 * commercial + 0.025 * relevance + 0.012 * rich, 0.01, 0.4)
    value = click * conversion * (1.8 + 0.8 * commercial) + 0.004 * conversation * tail
    abandonment = np.clip(
        0.08
        + 0.026 * two * (1 - commercial)
        + 0.025 * top * (relevance < 0.35)
        + 0.009 * conversation
        - 0.008 * inline,
        0.01,
        0.5,
    )
    utility = np.clip(
        0.62
        + 0.16 * context["prior_engagement"].to_numpy()
        + 0.04 * relevance
        - 0.038 * two * (1 - commercial)
        - 0.025 * top * (relevance < 0.35)
        + 0.012 * conversation * tail,
        0,
        1,
    )
    reformulation = np.clip(
        0.16 + 0.05 * (1 - relevance) + 0.018 * two - 0.012 * conversation * tail, 0.01, 0.5
    )
    organic = np.clip(
        0.55 + 0.12 * (1 - commercial) - 0.035 * two - 0.018 * top + 0.01 * inline, 0, 1
    )
    return {
        "click_probability": click,
        "conversion_probability": conversion,
        "advertiser_value_mean": value,
        "abandonment_probability": abandonment,
        "user_utility_mean": utility,
        "reformulation_probability": reformulation,
        "organic_engagement_mean": organic,
        "low_relevance_exposure": ((relevance < 0.35) & (top == 1)).astype(float),
        "ad_density": np.full(len(context), 1 + two),
    }


def run_observable_experiment(
    n: int,
    seed: int,
    experiment_id: str,
    base_context: pd.DataFrame | None = None,
    user_prefix: str = "",
) -> pd.DataFrame:
    if base_context is None:
        context = generate_context(n, seed)
    else:
        required = {
            "query_cluster",
            "commercial_intent",
            "relevance",
            "tail_query",
            "returning_user",
            "prior_exposure",
            "historical_pctr",
            "prior_engagement",
            "prior_advertiser_value",
        }
        missing = sorted(required - set(base_context.columns))
        if missing:
            raise ValueError(f"base context is missing pre-treatment fields: {missing}")
        context_rng = np.random.default_rng(seed)
        indices = context_rng.integers(0, len(base_context), n)
        context = base_context.iloc[indices].reset_index(drop=True).copy()
        users = context_rng.integers(1, max(2000, n // 3), n)
        context["sim_user_id"] = [f"{user_prefix}u{x}" for x in users]
    rng = np.random.default_rng(seed + 17)
    context["treatment"] = [treatment_for(experiment_id, x) for x in context["sim_user_id"]]
    context["jackknife_bucket"] = [
        jackknife_bucket(experiment_id, x) for x in context["sim_user_id"]
    ]
    context["policy_partition"] = [
        policy_partition(experiment_id, x) for x in context["sim_user_id"]
    ]
    context["week"] = [1 + int(x) for x in rng.integers(0, 6, n)]
    outputs = {
        name: np.empty(n)
        for name in [
            "click",
            "conversion",
            "advertiser_value",
            "abandonment",
            "user_utility",
            "reformulation",
            "organic_engagement",
            "low_relevance_exposure",
            "ad_density",
        ]
    }
    for treatment in context["treatment"].unique():
        mask = context["treatment"].eq(treatment).to_numpy()
        means = potential_means(context.loc[mask], treatment)
        clicks = rng.binomial(1, means["click_probability"])
        conversions = clicks * rng.binomial(1, means["conversion_probability"])
        outputs["click"][mask] = clicks
        outputs["conversion"][mask] = conversions
        outputs["advertiser_value"][mask] = conversions * rng.gamma(
            2,
            np.maximum(
                means["advertiser_value_mean"]
                / np.maximum(
                    means["click_probability"] * means["conversion_probability"] * 2, 1e-4
                ),
                1e-4,
            ),
        )
        outputs["abandonment"][mask] = rng.binomial(1, means["abandonment_probability"])
        outputs["user_utility"][mask] = np.clip(
            means["user_utility_mean"] + rng.normal(0, 0.12, mask.sum()), 0, 1
        )
        outputs["reformulation"][mask] = rng.binomial(1, means["reformulation_probability"])
        outputs["organic_engagement"][mask] = np.clip(
            means["organic_engagement_mean"] + rng.normal(0, 0.15, mask.sum()), 0, 1
        )
        outputs["low_relevance_exposure"][mask] = means["low_relevance_exposure"]
        outputs["ad_density"][mask] = means["ad_density"]
    for name, values in outputs.items():
        context[name] = values
    return context
