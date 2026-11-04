from __future__ import annotations

import pandas as pd
import pytest

from aix_page.models.baselines import FEATURE_COLUMNS, records


def test_feature_records_preserve_integer_ids_and_column_order():
    large_id = 2**53 + 1
    frame = pd.DataFrame({column: [large_id] for column in FEATURE_COLUMNS})
    frame["normalized_position"] = 0.25
    frame["hist_ctr"] = 0.125
    expected = {f"{column}={large_id}": 1.0 for column in FEATURE_COLUMNS}
    expected.update(normalized_position=0.25, hist_ctr=0.125)
    assert records(frame[list(reversed(frame.columns))]) == [expected]


def test_empty_feature_records():
    assert records(pd.DataFrame()) == []


def test_duplicate_feature_columns_are_rejected():
    with pytest.raises(ValueError, match="unique column"):
        records(pd.DataFrame([[1, 2]], columns=["Position", "Position"]))
