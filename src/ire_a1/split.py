"""Q1 step 3: temporal train/val/test split -- never random for interaction data.

Cutoffs are chosen as quantiles of the impression timestamp distribution (not min/max
arithmetic), so they're robust to a handful of outlier timestamps and adapt automatically to
each dataset/tier's actual time range, while still matching the PDF's "last N days as test,
preceding M days as validation" recipe -- test_frac/val_frac are just that recipe expressed as
fractions of the impression volume instead of a fixed day count.
"""
import polars as pl


def temporal_split(
    impressions: pl.DataFrame,
    time_col: str = "timestamp",
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> dict[str, pl.DataFrame]:
    if not 0 < test_frac < 1 or not 0 < val_frac < 1 or test_frac + val_frac >= 1:
        raise ValueError("test_frac and val_frac must each be in (0, 1) and sum to < 1")

    ts_sorted = impressions.get_column(time_col).sort()
    n = ts_sorted.len()
    test_cut = ts_sorted[int(n * (1 - test_frac))]
    val_cut = ts_sorted[int(n * (1 - test_frac - val_frac))]

    train = impressions.filter(pl.col(time_col) < val_cut)
    val = impressions.filter((pl.col(time_col) >= val_cut) & (pl.col(time_col) < test_cut))
    test = impressions.filter(pl.col(time_col) >= test_cut)

    _assert_no_leakage(train, val, test, time_col)
    return {"train": train, "val": val, "test": test}


def _assert_no_leakage(train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame, time_col: str) -> None:
    """Anti-gaming (Q9) starts here: a temporal split that isn't strictly ordered is a leakage
    bug, so fail loudly at split time rather than silently producing a too-good offline score."""
    train_max = train.get_column(time_col).max() if train.height else None
    val_min = val.get_column(time_col).min() if val.height else None
    val_max = val.get_column(time_col).max() if val.height else None
    test_min = test.get_column(time_col).min() if test.height else None

    if train_max is not None and val_min is not None:
        assert train_max <= val_min, f"leakage: train max {train_max} > val min {val_min}"
    if val_max is not None and test_min is not None:
        assert val_max <= test_min, f"leakage: val max {val_max} > test min {test_min}"
