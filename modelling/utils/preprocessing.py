"""Generic data preprocessing utilities."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def clamp_predictions(preds: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Clamp predictions to a positive floor."""
    return np.maximum(preds, floor)


def cap_target(
    df: pd.DataFrame,
    target_col: str,
    cap_percentile: float = 99.5,
    cap_value: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """Cap (winsorise) the target column.

    Creates a new column ``{target_col}_CAPPED`` with values clipped
    to the given percentile or hard cap.

    Returns:
        Tuple of (df_with_capped_col, actual_cap_value).
    """
    df = df.copy()
    capped_col = f"{target_col}_CAPPED"

    if cap_value is not None:
        actual_cap = cap_value
    else:
        actual_cap = float(np.percentile(df[target_col].dropna(), cap_percentile))

    df[capped_col] = df[target_col].clip(upper=actual_cap)
    n_capped = int((df[target_col] > actual_cap).sum())
    log.info(
        "  Target capped: %s → %s (cap=%.2f, %d rows affected)",
        target_col,
        capped_col,
        actual_cap,
        n_capped,
    )
    return df, actual_cap


def load_csv_with_split(
    input_path: str,
    target_col: str,
    split_col: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None,
    test_fraction: float = 0.2,
    seed: int = 42,
    quick: bool = False,
    quick_n: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV and split into train/test DataFrames.

    If split_col is provided, uses it to partition (TRAIN/TEST values).
    Otherwise, does a random split.

    Returns:
        Tuple of (train_df, test_df).
    """
    log.info("  Loading data from %s ...", input_path)
    df = pd.read_csv(input_path)
    log.info("  Loaded %d rows, %d columns", len(df), len(df.columns))

    # Drop excluded columns
    if exclude_cols:
        cols_to_drop = [c for c in exclude_cols if c in df.columns]
        # Also drop any *_IMPUTED columns
        imputed_cols = [c for c in df.columns if c.endswith("_IMPUTED")]
        cols_to_drop = list(set(cols_to_drop + imputed_cols))
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors="ignore")
            log.info("  Dropped %d excluded columns", len(cols_to_drop))

    # Split
    if split_col and split_col in df.columns:
        train_df = df[df[split_col] == "TRAIN"].drop(columns=[split_col]).copy()
        test_df = df[df[split_col] == "TEST"].drop(columns=[split_col]).copy()
        log.info(
            "  Split by '%s' — Train: %d, Test: %d",
            split_col,
            len(train_df),
            len(test_df),
        )
    else:
        if split_col:
            log.warning(
                "  Split column '%s' not found — using random %.0f%% split",
                split_col,
                (1 - test_fraction) * 100,
            )
        rng = np.random.RandomState(seed)
        mask = rng.rand(len(df)) < (1 - test_fraction)
        train_df = df[mask].copy()
        test_df = df[~mask].copy()
        log.info(
            "  Random split — Train: %d, Test: %d", len(train_df), len(test_df)
        )

    if quick:
        train_df = train_df.head(quick_n)
        log.info("  Quick mode — subsampled train to %d rows", len(train_df))

    return train_df, test_df
