import ast
from pathlib import Path

import numpy as np

from aix_page.constants import CONTROL
from aix_page.policy.learning import dr_value, fit_outcomes, ope_scores, policies
from aix_page.simulator.observable_generator import run_observable_experiment


def test_oracle_firewall_static_imports() -> None:
    root = Path(__file__).resolve().parents[2] / "src/aix_page"
    for folder in ["policy", "causal"]:
        for path in (root / folder).glob("*.py"):
            tree = ast.parse(path.read_text())
            assert not any(
                isinstance(node, ast.ImportFrom)
                and node.module
                and "oracle_evaluator" in node.module
                for node in ast.walk(tree)
            )


def test_hard_constraints_and_ope_toy_truth() -> None:
    data = run_observable_experiment(12000, 7, "test")
    discovery = data[data.policy_partition.eq("policy_discovery")]
    locked = data[data.policy_partition.eq("policy_locked_test")]
    value = fit_outcomes(discovery, "advertiser_value")
    utility = fit_outcomes(discovery, "user_utility")
    abandon = fit_outcomes(discovery, "abandonment")
    choice = policies(locked, value, utility, abandon, CONTROL)["P6_AIX_constrained"]
    estimate, se = dr_value(locked, choice, value, "advertiser_value")
    assert np.isfinite(estimate) and se >= 0 and len(choice) == len(locked)
    scores = ope_scores(locked, choice, value, "advertiser_value")
    assert set(scores) == {"ips", "snips_numerator", "snips_denominator", "dr"}
    assert np.isfinite(scores["dr"]).all()
