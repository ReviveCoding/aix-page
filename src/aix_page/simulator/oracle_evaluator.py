"""Sealed oracle evaluation surface. Import only during P16."""

from __future__ import annotations

import pandas as pd

from aix_page.constants import TREATMENTS
from aix_page.simulator.observable_generator import potential_means


def truth_table(context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for treatment in TREATMENTS:
        means = potential_means(context, treatment)
        block = pd.DataFrame(
            {"sim_user_id": context["sim_user_id"], "treatment": treatment, **means}
        )
        rows.append(block)
    return pd.concat(rows, ignore_index=True)
