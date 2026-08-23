from datetime import datetime, timedelta

import polars as pl
import pytest

from ire_a1.split import temporal_split


def _synthetic_impressions(n: int = 1000) -> pl.DataFrame:
    base = datetime(2024, 1, 1)
    return pl.DataFrame({
        "impression_id": [str(i) for i in range(n)],
        "timestamp": [base + timedelta(minutes=i) for i in range(n)],
    })


def test_split_is_strictly_ordered_in_time():
    """The core anti-leakage property (Q9): every train timestamp must be <= every val
    timestamp, and every val timestamp <= every test timestamp."""
    df = _synthetic_impressions()
    splits = temporal_split(df, test_frac=0.2, val_frac=0.2)

    assert splits["train"].height + splits["val"].height + splits["test"].height == df.height
    assert splits["train"]["timestamp"].max() <= splits["val"]["timestamp"].min()
    assert splits["val"]["timestamp"].max() <= splits["test"]["timestamp"].min()


def test_split_fractions_are_approximately_respected():
    df = _synthetic_impressions(n=10_000)
    splits = temporal_split(df, test_frac=0.15, val_frac=0.15)

    assert abs(splits["test"].height / df.height - 0.15) < 0.02
    assert abs(splits["val"].height / df.height - 0.15) < 0.02
    assert abs(splits["train"].height / df.height - 0.70) < 0.02


def test_split_rejects_bad_fractions():
    df = _synthetic_impressions(n=100)
    with pytest.raises(ValueError):
        temporal_split(df, test_frac=0.6, val_frac=0.6)  # sums to >= 1
    with pytest.raises(ValueError):
        temporal_split(df, test_frac=0, val_frac=0.1)
