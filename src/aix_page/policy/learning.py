"""Cross-partition policy ladder and known-propensity OPE."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from aix_page.constants import CONTROL, TREATMENTS

FEATURES = [
    "commercial_intent",
    "relevance",
    "tail_query",
    "returning_user",
    "prior_exposure",
    "historical_pctr",
    "prior_engagement",
    "prior_advertiser_value",
]


def fit_outcomes(discovery: pd.DataFrame, outcome: str) -> dict[str, GradientBoostingRegressor]:
    models = {}
    for treatment in TREATMENTS:
        arm = discovery[discovery.treatment.eq(treatment)]
        model = GradientBoostingRegressor(
            n_estimators=40, max_depth=2, learning_rate=0.06, loss="huber", random_state=17
        )
        model.fit(arm[FEATURES], arm[outcome])
        models[treatment] = model
    return models


def prediction_matrix(
    models: dict[str, GradientBoostingRegressor], frame: pd.DataFrame
) -> np.ndarray:
    return np.column_stack([models[t].predict(frame[FEATURES]) for t in TREATMENTS])


def policies(
    frame: pd.DataFrame,
    value_models: dict[str, GradientBoostingRegressor],
    utility_models: dict[str, GradientBoostingRegressor],
    abandonment_models: dict[str, GradientBoostingRegressor],
    best_static: str,
    organic_models: dict[str, GradientBoostingRegressor] | None = None,
    global_winner: str | None = None,
) -> dict[str, np.ndarray]:
    value = prediction_matrix(value_models, frame)
    utility = prediction_matrix(utility_models, frame)
    abandon = prediction_matrix(abandonment_models, frame)
    ctr_score = np.tile(frame["historical_pctr"].to_numpy()[:, None], (1, len(TREATMENTS)))
    for j, treatment in enumerate(TREATMENTS):
        fmt, pos, density = treatment.split("__")
        ctr_score[:, j] += (
            0.01 * (pos == "top") + 0.012 * (fmt == "rich_product") + 0.007 * (density == "two")
        )
    control_index = TREATMENTS.index(CONTROL)
    safe = (utility >= utility[:, [control_index]] - 0.015) & (
        abandon <= abandon[:, [control_index]] + 0.010
    )
    if organic_models is not None:
        organic = prediction_matrix(organic_models, frame)
        safe &= organic >= organic[:, [control_index]] - 0.015
    relevance_safe = np.ones((len(frame), len(TREATMENTS)), dtype=bool)
    contextual_density_safe = np.ones_like(relevance_safe)
    for j, treatment in enumerate(TREATMENTS):
        _, position, density = treatment.split("__")
        if position == "top":
            relevance_safe[:, j] = frame["relevance"].to_numpy() >= 0.35
        if density == "two":
            contextual_density_safe[:, j] = (frame["commercial_intent"].to_numpy() == 1) & (
                frame["relevance"].to_numpy() >= 0.45
            )
    safe &= relevance_safe & contextual_density_safe
    safe[:, TREATMENTS.index(CONTROL)] = True
    constrained = np.where(safe, value, -np.inf)
    rel_ctr = np.where(relevance_safe, ctr_score, -np.inf)
    return {
        "P0_Control": np.full(len(frame), CONTROL),
        "P1_Best_safe_static": np.full(len(frame), best_static),
        "P2_Global_AB_winner": np.full(len(frame), global_winner or best_static),
        "P3_CTR_greedy": np.asarray(TREATMENTS)[np.argmax(ctr_score, axis=1)],
        "P4_Relevance_constrained_CTR": np.asarray(TREATMENTS)[np.argmax(rel_ctr, axis=1)],
        "P5_Unconstrained_CATE": np.asarray(TREATMENTS)[np.argmax(value, axis=1)],
        "P6_AIX_constrained": np.asarray(TREATMENTS)[np.argmax(constrained, axis=1)],
    }


def ope_scores(
    frame: pd.DataFrame,
    choices: np.ndarray,
    models: dict[str, GradientBoostingRegressor],
    outcome: str,
) -> dict[str, np.ndarray]:
    """Known-propensity IPS, SNIPS contributions, and DR scores."""
    matrix = prediction_matrix(models, frame)
    return ope_scores_from_matrix(frame, choices, matrix, outcome)


def ope_scores_from_matrix(
    frame: pd.DataFrame,
    choices: np.ndarray,
    matrix: np.ndarray,
    outcome: str,
) -> dict[str, np.ndarray]:
    """OPE scores using a precomputed all-treatment outcome matrix."""
    if matrix.shape != (len(frame), len(TREATMENTS)):
        raise ValueError("outcome prediction matrix has an invalid shape")
    indices = np.asarray([TREATMENTS.index(str(choice)) for choice in choices])
    selected = matrix[np.arange(len(frame)), indices]
    observed_indices = np.asarray([TREATMENTS.index(str(t)) for t in frame["treatment"]])
    observed_model = matrix[np.arange(len(frame)), observed_indices]
    matched = frame["treatment"].to_numpy() == choices
    inverse = matched.astype(float) * len(TREATMENTS)
    observed = frame[outcome].to_numpy(float)
    return {
        "ips": inverse * observed,
        "snips_numerator": inverse * observed,
        "snips_denominator": inverse,
        "dr": selected + inverse * (observed - observed_model),
    }


def dr_value(
    frame: pd.DataFrame,
    choices: np.ndarray,
    models: dict[str, GradientBoostingRegressor],
    outcome: str,
) -> tuple[float, float]:
    scores = ope_scores(frame, choices, models, outcome)["dr"]
    user_means = (
        pd.DataFrame({"user": frame["sim_user_id"], "score": scores}).groupby("user").score.mean()
    )
    return float(scores.mean()), float(user_means.std(ddof=1) / np.sqrt(len(user_means)))
