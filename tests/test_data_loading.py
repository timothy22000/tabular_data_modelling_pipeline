"""Tests for data loading and target capping utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modelling.utils.preprocessing import cap_target, load_csv_with_split


# ---------------------------------------------------------------------------
# load_csv_with_split
# ---------------------------------------------------------------------------

class TestLoadCsvWithSplit:
    def test_split_col_partitions_correctly(
        self, synthetic_df: pd.DataFrame, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """When split_col is present, rows should be split by TRAIN/TEST values."""
        df = synthetic_df.copy()
        rng = np.random.default_rng(0)
        split_vals = rng.choice(["TRAIN", "TEST"], size=len(df), p=[0.8, 0.2])
        df["split"] = split_vals

        csv_path = tmp_path / "split_data.csv"
        df.to_csv(csv_path, index=False)

        train_df, test_df = load_csv_with_split(
            str(csv_path), target_col="target", split_col="split"
        )

        expected_train = int((split_vals == "TRAIN").sum())
        expected_test = int((split_vals == "TEST").sum())

        assert len(train_df) == expected_train
        assert len(test_df) == expected_test
        # split column must be dropped from both outputs
        assert "split" not in train_df.columns
        assert "split" not in test_df.columns

    def test_random_split_produces_correct_size(
        self, synthetic_csv_path: str
    ) -> None:
        """When split_col is None, a random 80/20 split should be used."""
        train_df, test_df = load_csv_with_split(
            synthetic_csv_path,
            target_col="target",
            split_col=None,
            test_fraction=0.2,
            seed=42,
        )

        total = len(train_df) + len(test_df)
        assert total == 100  # synthetic_df has 100 rows

        # Allow a small tolerance around 80 / 20
        assert len(train_df) >= 70
        assert len(test_df) >= 10

    def test_random_split_is_reproducible(self, synthetic_csv_path: str) -> None:
        """Same seed must produce identical splits on repeated calls."""
        train1, test1 = load_csv_with_split(
            synthetic_csv_path, target_col="target", seed=7
        )
        train2, test2 = load_csv_with_split(
            synthetic_csv_path, target_col="target", seed=7
        )

        pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))

    def test_missing_split_col_falls_back_to_random(
        self, synthetic_csv_path: str
    ) -> None:
        """When split_col is specified but absent from the CSV, a random split is used."""
        train_df, test_df = load_csv_with_split(
            synthetic_csv_path,
            target_col="target",
            split_col="nonexistent_col",
            seed=42,
        )
        assert len(train_df) + len(test_df) == 100

    def test_quick_mode_subsamples_train(self, synthetic_csv_path: str) -> None:
        """quick=True must limit training rows to quick_n."""
        train_df, _ = load_csv_with_split(
            synthetic_csv_path,
            target_col="target",
            quick=True,
            quick_n=20,
            seed=42,
        )
        assert len(train_df) <= 20

    def test_exclude_cols_removes_columns(
        self, synthetic_df: pd.DataFrame, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Columns listed in exclude_cols must not appear in either output."""
        csv_path = tmp_path / "data_excl.csv"
        synthetic_df.to_csv(csv_path, index=False)

        train_df, test_df = load_csv_with_split(
            str(csv_path),
            target_col="target",
            exclude_cols=["feat_c"],
        )

        assert "feat_c" not in train_df.columns
        assert "feat_c" not in test_df.columns


# ---------------------------------------------------------------------------
# cap_target
# ---------------------------------------------------------------------------

class TestCapTarget:
    def test_percentile_cap_creates_capped_column(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """cap_target must create a '{target_col}_CAPPED' column."""
        df_out, cap_val = cap_target(synthetic_df, "target", cap_percentile=99.5)
        assert "target_CAPPED" in df_out.columns

    def test_percentile_cap_clips_values(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """All values in the capped column must be <= the returned cap_value."""
        df_out, cap_val = cap_target(synthetic_df, "target", cap_percentile=80.0)
        assert (df_out["target_CAPPED"] <= cap_val).all()

    def test_hard_cap_overrides_percentile(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """When cap_value is provided explicitly, it must be used as the cap."""
        hard_cap = 50.0
        df_out, cap_val = cap_target(
            synthetic_df, "target", cap_percentile=99.5, cap_value=hard_cap
        )
        assert cap_val == hard_cap
        assert (df_out["target_CAPPED"] <= hard_cap).all()

    def test_original_column_unchanged(self, synthetic_df: pd.DataFrame) -> None:
        """The original target column must not be modified."""
        original_values = synthetic_df["target"].copy()
        df_out, _ = cap_target(synthetic_df, "target", cap_percentile=50.0)
        pd.testing.assert_series_equal(
            df_out["target"].reset_index(drop=True),
            original_values.reset_index(drop=True),
        )

    def test_returns_tuple_of_df_and_float(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """cap_target must return a (DataFrame, float) tuple."""
        result = cap_target(synthetic_df, "target")
        assert isinstance(result, tuple)
        assert len(result) == 2
        df_out, cap_val = result
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(cap_val, float)

    def test_no_modification_of_input_df(self, synthetic_df: pd.DataFrame) -> None:
        """cap_target must not mutate the input DataFrame."""
        original_cols = list(synthetic_df.columns)
        cap_target(synthetic_df, "target")
        assert list(synthetic_df.columns) == original_cols
