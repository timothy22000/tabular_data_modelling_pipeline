"""Tests for training utilities: _get_monotone_cont_indices and TabularDataset."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pytest

from modelling.training import _get_monotone_cont_indices


# ---------------------------------------------------------------------------
# _get_monotone_cont_indices
# ---------------------------------------------------------------------------

class TestGetMonotoneContinuousIndices:
    def test_empty_constraints_returns_empty_lists(self) -> None:
        """No constraints -> both returned lists must be empty."""
        indices, directions = _get_monotone_cont_indices(
            ["feat_a", "feat_b", "feat_c"], monotone_constraints={}
        )
        assert indices == []
        assert directions == []

    def test_none_constraints_returns_empty_lists(self) -> None:
        """None constraints -> both returned lists must be empty."""
        indices, directions = _get_monotone_cont_indices(
            ["feat_a", "feat_b"], monotone_constraints=None
        )
        assert indices == []
        assert directions == []

    def test_single_positive_constraint(self) -> None:
        """A single +1 constraint on the second feature returns index=1, direction=+1."""
        indices, directions = _get_monotone_cont_indices(
            ["feat_a", "feat_b", "feat_c"],
            monotone_constraints={"feat_b": 1},
        )
        assert indices == [1]
        assert directions == [1]

    def test_single_negative_constraint(self) -> None:
        """A single -1 constraint on the first feature returns index=0, direction=-1."""
        indices, directions = _get_monotone_cont_indices(
            ["feat_a", "feat_b"],
            monotone_constraints={"feat_a": -1},
        )
        assert indices == [0]
        assert directions == [-1]

    def test_multiple_constraints(self) -> None:
        """Multiple constraints must be discovered in feature-list order."""
        feature_names = ["feat_a", "feat_b", "feat_c"]
        constraints = {"feat_a": 1, "feat_c": -1}
        indices, directions = _get_monotone_cont_indices(feature_names, constraints)

        assert indices == [0, 2]
        assert directions == [1, -1]

    def test_constraint_on_absent_feature_is_ignored(self) -> None:
        """A constraint on a feature not in the continuous list must be ignored."""
        indices, directions = _get_monotone_cont_indices(
            ["feat_a", "feat_b"],
            monotone_constraints={"feat_z": 1},  # feat_z not in list
        )
        assert indices == []
        assert directions == []

    def test_zero_direction_is_excluded(self) -> None:
        """A constraint with direction=0 must be treated as no constraint."""
        indices, directions = _get_monotone_cont_indices(
            ["feat_a", "feat_b"],
            monotone_constraints={"feat_a": 0},
        )
        assert indices == []
        assert directions == []

    def test_returns_two_lists(self) -> None:
        """Function must always return a tuple of exactly two lists."""
        result = _get_monotone_cont_indices(["feat_a"], {})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_indices_and_directions_same_length(self) -> None:
        """Returned indices and directions must have equal length."""
        indices, directions = _get_monotone_cont_indices(
            ["a", "b", "c", "d"],
            {"a": 1, "c": -1, "d": 1},
        )
        assert len(indices) == len(directions)


# ---------------------------------------------------------------------------
# TabularDataset
# ---------------------------------------------------------------------------

class TestTabularDataset:
    """Tests for the PyTorch-backed TabularDataset class."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.fixture()
    def dataset(self):
        """Return a small TabularDataset instance."""
        import torch
        from modelling.data import TabularDataset

        rng = np.random.default_rng(99)
        n = 20
        x_cont = rng.standard_normal((n, 3)).astype(np.float32)
        x_cat = rng.integers(0, 3, (n, 2)).astype(np.int64)
        y = (rng.uniform(10, 100, n)).astype(np.float32)
        return TabularDataset(x_cont=x_cont, x_cat=x_cat, y=y)

    def test_len_equals_number_of_rows(self, dataset) -> None:
        """__len__ must equal the number of rows in the dataset."""
        assert len(dataset) == 20

    def test_getitem_returns_five_tensors(self, dataset) -> None:
        """__getitem__ must return a tuple of exactly 5 tensors."""
        import torch

        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 5
        for t in item:
            assert isinstance(t, torch.Tensor)

    def test_getitem_correct_shapes(self, dataset) -> None:
        """Returned tensors must have the expected shapes."""
        x_cont, x_cat, glm_pred, gbm_pred, y = dataset[0]
        assert x_cont.shape == (3,)
        assert x_cat.shape == (2,)
        assert glm_pred.shape == ()  # scalar
        assert gbm_pred.shape == ()
        assert y.shape == ()

    def test_glm_preds_default_to_ones(self) -> None:
        """When glm_preds is not provided, it must default to 1.0."""
        from modelling.data import TabularDataset

        rng = np.random.default_rng(0)
        x_cont = rng.standard_normal((5, 2)).astype(np.float32)
        x_cat = np.zeros((5, 1), dtype=np.int64)
        y = np.ones(5, dtype=np.float32) * 10.0

        ds = TabularDataset(x_cont=x_cont, x_cat=x_cat, y=y)
        _, _, glm_pred, gbm_pred, _ = ds[0]
        assert glm_pred.item() == 1.0
        assert gbm_pred.item() == 1.0

    def test_custom_glm_and_gbm_preds_stored(self) -> None:
        """Provided glm/gbm predictions must be accessible via __getitem__."""
        from modelling.data import TabularDataset

        rng = np.random.default_rng(0)
        n = 5
        x_cont = rng.standard_normal((n, 2)).astype(np.float32)
        x_cat = np.zeros((n, 1), dtype=np.int64)
        y = np.ones(n, dtype=np.float32) * 10.0
        glm_preds = np.arange(1, n + 1, dtype=np.float32) * 5.0
        gbm_preds = np.arange(1, n + 1, dtype=np.float32) * 7.0

        ds = TabularDataset(x_cont=x_cont, x_cat=x_cat, y=y,
                            glm_preds=glm_preds, gbm_preds=gbm_preds)

        _, _, glm_val, gbm_val, _ = ds[2]
        assert abs(glm_val.item() - 15.0) < 1e-5  # index 2 -> 3 * 5.0
        assert abs(gbm_val.item() - 21.0) < 1e-5  # index 2 -> 3 * 7.0

    def test_indexing_last_element(self, dataset) -> None:
        """Accessing the last index must not raise an IndexError."""
        last_idx = len(dataset) - 1
        item = dataset[last_idx]
        assert len(item) == 5
