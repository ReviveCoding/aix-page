"""Stable experiment and analysis partition assignment."""

from __future__ import annotations

from aix_page.constants import TREATMENTS
from aix_page.utils.hashing import stable_mod


def treatment_for(experiment_id: str, sim_user_id: str) -> str:
    return TREATMENTS[stable_mod(len(TREATMENTS), experiment_id, sim_user_id)]


def jackknife_bucket(experiment_id: str, sim_user_id: str) -> int:
    return stable_mod(20, experiment_id, sim_user_id, "jackknife")


def policy_partition(experiment_id: str, sim_user_id: str) -> str:
    bucket = stable_mod(100, experiment_id, sim_user_id, "policy")
    return (
        "policy_discovery"
        if bucket < 60
        else "policy_selection"
        if bucket < 80
        else "policy_locked_test"
    )
