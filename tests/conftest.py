"""Shared pytest fixtures for the tabular modelling pipeline test suite."""
from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so top-level modules are importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest

from dataset_config import DatasetConfig
from modelling.config import DLConfig


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Return a 100-row synthetic DataFrame for pipeline testing.

    Columns:
        feat_a  -- uniform(0, 100)
        feat_b  -- normal(mu=50, sigma=10)
        feat_c  -- uniform(1, 10)
        cat_x   -- categorical with values A / B / C
        cat_y   -- categorical with values X / Y
        target  -- abs(feat_a*0.5 + feat_b*0.3 + noise) + 10  (always > 10)
    """
    rng = np.random.default_rng(42)
    n = 100

    feat_a = rng.uniform(0, 100, n)
    feat_b = rng.normal(50, 10, n)
    feat_c = rng.uniform(1, 10, n)
    cat_x = rng.choice(["A", "B", "C"], n)
    cat_y = rng.choice(["X", "Y"], n)
    noise = rng.normal(0, 5, n)
    target = np.abs(feat_a * 0.5 + feat_b * 0.3 + noise) + 10.0

    return pd.DataFrame(
        {
            "feat_a": feat_a.astype(np.float64),
            "feat_b": feat_b.astype(np.float64),
            "feat_c": feat_c.astype(np.float64),
            "cat_x": cat_x,
            "cat_y": cat_y,
            "target": target.astype(np.float64),
        }
    )


@pytest.fixture(scope="session")
def sample_config() -> DatasetConfig:
    """Return a DatasetConfig configured for the synthetic dataset."""
    return DatasetConfig(
        target_col="target",
        continuous_features=["feat_a", "feat_b", "feat_c"],
        categorical_features=["cat_x", "cat_y"],
        family="gamma",
        link="log",
        prediction_floor=1.0,
        cap_percentile=99.5,
    )


@pytest.fixture(scope="session")
def sample_dl_config(sample_config: DatasetConfig) -> DLConfig:
    """Return a DLConfig with quick / skip flags set for fast test runs."""
    return DLConfig(
        dataset=sample_config,
        quick=True,
        skip_tuning=True,
        skip_interpretability=True,
        epochs=2,
        patience=2,
        batch_size=32,
        n_ensemble=1,
        architectures=["cann", "ft_transformer", "tabm", "cann_gbm", "localglmnet", "drn"],
    )


@pytest.fixture(scope="session")
def synthetic_csv_path(synthetic_df: pd.DataFrame, tmp_path_factory) -> str:
    """Write the synthetic DataFrame to a temporary CSV and return the path."""
    tmp_dir = tmp_path_factory.mktemp("data")
    csv_path = tmp_dir / "synthetic.csv"
    synthetic_df.to_csv(csv_path, index=False)
    return str(csv_path)
