"""Chronological dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from mev_dataset.config import SplitConfig


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def compute_split_boundaries(timestamps: Iterable[pd.Timestamp], splits: SplitConfig) -> SplitBoundaries:
    unique = pd.Index(pd.to_datetime(pd.Series(list(timestamps)), utc=True).dropna().drop_duplicates()).sort_values()
    if unique.empty:
        raise ValueError("Cannot compute split boundaries on an empty timestamp set")
    n = len(unique)
    train_end_idx = max(int(n * splits.train) - 1, 0)
    validation_end_idx = max(int(n * (splits.train + splits.validation)) - 1, train_end_idx)
    validation_end_idx = min(validation_end_idx, n - 1)
    return SplitBoundaries(train_end=unique[train_end_idx], validation_end=unique[validation_end_idx])


def assign_splits(df: pd.DataFrame, boundaries: SplitBoundaries, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True)
    out["split"] = "test"
    out.loc[out[timestamp_col] <= boundaries.train_end, "split"] = "train"
    out.loc[
        (out[timestamp_col] > boundaries.train_end) & (out[timestamp_col] <= boundaries.validation_end),
        "split",
    ] = "validation"
    return out


def build_split_assignments(timestamps: Iterable[pd.Timestamp], splits: SplitConfig) -> tuple[pd.DataFrame, SplitBoundaries]:
    boundaries = compute_split_boundaries(timestamps, splits)
    assignments = pd.DataFrame({"timestamp": pd.Index(sorted(set(pd.to_datetime(list(timestamps), utc=True))))})
    assignments = assign_splits(assignments, boundaries)
    return assignments, boundaries


def write_split_manifest(
    output_dir: str | Path,
    assignments: pd.DataFrame,
    boundaries: SplitBoundaries,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(output_path / "split_assignments.parquet", index=False)
    manifest = {
        "train_end": boundaries.train_end.isoformat(),
        "validation_end": boundaries.validation_end.isoformat(),
        "counts": assignments["split"].value_counts().sort_index().to_dict(),
    }
    (output_path / "split_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
