"""Tests for DatasetConfig and auto_detect_features."""
from __future__ import annotations

import pandas as pd
import pytest

from dataset_config import DatasetConfig, auto_detect_features


class TestDatasetConfigValidation:
    def test_valid_config_passes(self, sample_config: DatasetConfig) -> None:
        """validate() must not raise for a properly constructed config."""
        sample_config.validate()  # should not raise

    def test_default_config_passes(self) -> None:
        """Default DatasetConfig (target_col='target') is valid."""
        DatasetConfig().validate()

    def test_missing_target_col_raises(self) -> None:
        """validate() must raise ValueError when target_col is empty."""
        cfg = DatasetConfig(target_col="")
        with pytest.raises(ValueError, match="target_col must be specified"):
            cfg.validate()

    def test_invalid_family_raises(self) -> None:
        """validate() must raise ValueError for an unsupported family."""
        cfg = DatasetConfig(family="poisson_xyz")
        with pytest.raises(ValueError, match="Unsupported family"):
            cfg.validate()

    def test_valid_families_pass(self) -> None:
        """All supported family strings must pass validation."""
        for family in ("gamma", "gaussian", "tweedie", "poisson"):
            DatasetConfig(family=family).validate()

    def test_invalid_link_raises(self) -> None:
        """validate() must raise ValueError for an unsupported link."""
        cfg = DatasetConfig(link="probit")
        with pytest.raises(ValueError, match="Unsupported link"):
            cfg.validate()

    def test_valid_links_pass(self) -> None:
        """Both supported link strings must pass validation."""
        for link in ("log", "identity"):
            DatasetConfig(link=link).validate()

    def test_zero_prediction_floor_raises(self) -> None:
        """validate() must raise ValueError when prediction_floor <= 0."""
        cfg = DatasetConfig(prediction_floor=0.0)
        with pytest.raises(ValueError, match="prediction_floor must be positive"):
            cfg.validate()

    def test_negative_prediction_floor_raises(self) -> None:
        cfg = DatasetConfig(prediction_floor=-1.0)
        with pytest.raises(ValueError, match="prediction_floor must be positive"):
            cfg.validate()

    def test_invalid_monotone_direction_raises(self) -> None:
        """validate() must raise ValueError for a monotone direction other than +1/-1."""
        cfg = DatasetConfig(monotone_constraints={"feat_a": 0})
        with pytest.raises(ValueError, match="must be \\+1 or -1"):
            cfg.validate()

    def test_valid_monotone_constraints_pass(self) -> None:
        """Both +1 and -1 monotone directions should pass validation."""
        cfg = DatasetConfig(monotone_constraints={"feat_a": 1, "feat_b": -1})
        cfg.validate()


class TestAutoDetectFeatures:
    def test_separates_numeric_and_string_columns(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """auto_detect_features must place numeric cols in continuous and
        string cols in categorical."""
        cfg = auto_detect_features(synthetic_df, target_col="target")

        assert set(cfg.continuous_features) == {"feat_a", "feat_b", "feat_c"}
        assert set(cfg.categorical_features) == {"cat_x", "cat_y"}

    def test_target_excluded_from_features(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """The target column must not appear in either feature list."""
        cfg = auto_detect_features(synthetic_df, target_col="target")

        assert "target" not in cfg.continuous_features
        assert "target" not in cfg.categorical_features

    def test_exclude_cols_respected(self, synthetic_df: pd.DataFrame) -> None:
        """Columns listed in exclude_cols must not appear in feature lists."""
        cfg = auto_detect_features(
            synthetic_df, target_col="target", exclude_cols=["feat_c"]
        )

        assert "feat_c" not in cfg.continuous_features
        assert "feat_c" not in cfg.categorical_features

    def test_returns_dataset_config(self, synthetic_df: pd.DataFrame) -> None:
        """auto_detect_features must return a DatasetConfig instance."""
        result = auto_detect_features(synthetic_df, target_col="target")
        assert isinstance(result, DatasetConfig)

    def test_target_col_set_correctly(self, synthetic_df: pd.DataFrame) -> None:
        """target_col on the returned config must match the argument."""
        cfg = auto_detect_features(synthetic_df, target_col="target")
        assert cfg.target_col == "target"

    def test_no_columns_in_both_lists(self, synthetic_df: pd.DataFrame) -> None:
        """No column should appear in both continuous and categorical."""
        cfg = auto_detect_features(synthetic_df, target_col="target")
        overlap = set(cfg.continuous_features) & set(cfg.categorical_features)
        assert overlap == set()
