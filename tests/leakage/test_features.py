import pandas as pd

from aix_page.features.pipeline import assign_partitions, crossfit_features
from aix_page.ingestion.synthetic import make_aggregated_fixture


def test_setting_groups_never_cross_partitions() -> None:
    data = make_aggregated_fixture(2000, 1)
    duplicate = pd.concat([data, data.iloc[:100]], ignore_index=True)
    split = assign_partitions(duplicate)
    assert split.groupby("setting_key").partition.nunique().max() == 1


def test_oof_singleton_does_not_leak_own_target() -> None:
    split = assign_partitions(make_aggregated_fixture(8000, 2))
    featured = crossfit_features(split)
    assert featured.filter(like="hist_").notna().all().all()
    assert all("Click" not in column for column in featured.filter(like="hist_").columns)
