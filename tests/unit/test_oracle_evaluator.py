import pandas as pd

from aix_page.constants import TREATMENTS
from aix_page.simulator.observable_generator import generate_context
from aix_page.simulator.oracle_evaluator import truth_table


def test_oracle_truth_table_has_every_treatment_per_context() -> None:
    context = generate_context(17, 41)
    truth = truth_table(context)
    assert len(truth) == len(context) * len(TREATMENTS)
    assert set(truth["treatment"]) == set(TREATMENTS)
    assert truth.groupby("sim_user_id")["treatment"].nunique().eq(len(TREATMENTS)).all()


def test_oracle_truth_values_have_valid_ranges() -> None:
    truth = truth_table(generate_context(11, 42))
    probability_columns = [
        "click_probability",
        "conversion_probability",
        "abandonment_probability",
        "reformulation_probability",
    ]
    for column in probability_columns:
        assert truth[column].between(0, 1).all()
    assert (truth["advertiser_value_mean"] >= 0).all()
    assert isinstance(truth, pd.DataFrame)
