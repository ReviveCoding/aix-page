import pandas as pd
import pytest

from aix_page.models.baselines import records, require_string_columns


def baseline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Position": [1],
            "Depth": [2],
            "AdID": [3],
            "AdvertiserID": [4],
            "DisplayURL": [5],
            "QueryID": [6],
            "normalized_position": [0.5],
            "hist_AdID_ctr": [0.125],
        }
    )


def test_require_string_columns_accepts_valid_frame() -> None:
    frame = baseline_frame()
    assert require_string_columns(frame) == list(frame.columns)


def test_require_string_columns_rejects_non_string_label() -> None:
    frame = baseline_frame()
    frame[7] = 1
    with pytest.raises(TypeError, match="require string column labels.*7.*int"):
        require_string_columns(frame)


def test_records_preserves_feature_contract() -> None:
    assert records(baseline_frame()) == [
        {
            "Position=1": 1.0,
            "Depth=2": 1.0,
            "AdID=3": 1.0,
            "AdvertiserID=4": 1.0,
            "DisplayURL=5": 1.0,
            "QueryID=6": 1.0,
            "normalized_position": 0.5,
            "hist_AdID_ctr": 0.125,
        }
    ]
