from datetime import UTC, datetime, timedelta

import pandas as pd

from mev_dataset.config import SplitConfig
from mev_dataset.split import assign_splits, build_split_assignments, compute_split_boundaries


def test_chronological_split_logic():
    timestamps = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(20)]
    splits = SplitConfig(train=0.70, validation=0.15, test=0.15)
    boundaries = compute_split_boundaries(timestamps, splits)
    assigned, returned_boundaries = build_split_assignments(timestamps, splits)

    assert boundaries == returned_boundaries
    assert assigned[assigned["split"] == "train"].shape[0] == 14
    assert assigned[assigned["split"] == "validation"].shape[0] == 3
    assert assigned[assigned["split"] == "test"].shape[0] == 3

    labelled = assign_splits(pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True)}), boundaries)
    assert labelled.iloc[0]["split"] == "train"
    assert labelled.iloc[-1]["split"] == "test"
