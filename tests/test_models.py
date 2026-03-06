"""Tests for DL model construction and forward passes.

Each architecture is tested for:
  - build_dl_model creates the correct model type.
  - Forward pass returns correct output shape.
  - Output is finite (no NaN / Inf).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from modelling.config import HAS_TORCH  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

from modelling.models import build_dl_model, get_default_dl_params  # noqa: E402
from modelling.models.cann import CANN  # noqa: E402
from modelling.models.cann_gbm import CANNGBM  # noqa: E402
from modelling.models.drn import DistributionalRefinementNetwork  # noqa: E402
from modelling.models.ft_transformer import FTTransformer  # noqa: E402
from modelling.models.localglmnet import LocalGLMnet  # noqa: E402
from modelling.models.tabm import TabMWrapper  # noqa: E402
from modelling.data import DLFeatureBundle  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BATCH = 16
_N_CONT = 3
_N_CAT = 2
_N_TRAIN = 80
_N_TEST = 20

# category_sizes: number of distinct categories INCLUDING the UNKNOWN (0) token
_CATEGORY_SIZES = [4, 3]  # cat_x: A/B/C + UNKNOWN=4; cat_y: X/Y + UNKNOWN=3
_EMBEDDING_DIMS = [2, 2]


def _make_bundle() -> DLFeatureBundle:
    """Return a minimal DLFeatureBundle suitable for model construction."""
    rng = np.random.default_rng(0)

    X_train_cont = rng.standard_normal((_N_TRAIN, _N_CONT)).astype(np.float32)
    X_test_cont = rng.standard_normal((_N_TEST, _N_CONT)).astype(np.float32)

    X_train_cat = rng.integers(0, 3, size=(_N_TRAIN, _N_CAT)).astype(np.int64)
    X_test_cat = rng.integers(0, 3, size=(_N_TEST, _N_CAT)).astype(np.int64)

    y_train = (rng.uniform(10, 200, _N_TRAIN)).astype(np.float32)
    y_test = (rng.uniform(10, 200, _N_TEST)).astype(np.float32)

    glm_train = np.ones(_N_TRAIN, dtype=np.float32) * 50.0
    glm_test = np.ones(_N_TEST, dtype=np.float32) * 50.0

    category_mappings: Dict[str, Dict[str, int]] = {
        "cat_x": {"UNKNOWN": 0, "A": 1, "B": 2, "C": 3},
        "cat_y": {"UNKNOWN": 0, "X": 1, "Y": 2},
    }

    return DLFeatureBundle(
        X_train_cont=X_train_cont,
        X_test_cont=X_test_cont,
        X_train_cat=X_train_cat,
        X_test_cat=X_test_cat,
        y_train=y_train,
        y_test=y_test,
        continuous_feature_names=["feat_a", "feat_b", "feat_c"],
        categorical_feature_names=["cat_x", "cat_y"],
        category_mappings=category_mappings,
        embedding_dims=_EMBEDDING_DIMS,
        cont_mean=np.zeros(_N_CONT, dtype=np.float32),
        cont_std=np.ones(_N_CONT, dtype=np.float32),
        glm_train_preds=glm_train,
        glm_test_preds=glm_test,
        catboost_train_pool=None,
        catboost_test_pool=None,
        train_df=None,  # type: ignore[arg-type]
        test_df=None,   # type: ignore[arg-type]
        gbm_train_preds=np.ones(_N_TRAIN, dtype=np.float32) * 50.0,
        gbm_test_preds=np.ones(_N_TEST, dtype=np.float32) * 50.0,
        glm_dispersion=1.0,
    )


def _make_batch() -> Tuple[
    "torch.Tensor", "torch.Tensor", "torch.Tensor"
]:
    """Return (x_cont, x_cat, glm_pred) mini-batch tensors."""
    rng = np.random.default_rng(1)
    x_cont = torch.tensor(
        rng.standard_normal((_BATCH, _N_CONT)), dtype=torch.float32
    )
    x_cat = torch.tensor(
        rng.integers(0, 3, size=(_BATCH, _N_CAT)), dtype=torch.long
    )
    glm_pred = torch.ones(_BATCH) * 50.0
    return x_cont, x_cat, glm_pred


@pytest.fixture(scope="module")
def bundle() -> DLFeatureBundle:
    return _make_bundle()


# ---------------------------------------------------------------------------
# Lightweight parameter sets (smaller than defaults for speed)
# ---------------------------------------------------------------------------

_SMALL_PARAMS: Dict[str, Dict] = {
    "cann": {"hidden_dims": [16, 8], "dropout": 0.0},
    "ft_transformer": {"d_model": 16, "n_heads": 2, "n_layers": 1, "dropout": 0.0, "ffn_factor": 2},
    "tabm": {"n_members": 2, "hidden_dims": [16, 8], "dropout": 0.0},
    "cann_gbm": {"hidden_dims": [16, 8], "dropout": 0.0},
    "localglmnet": {"hidden_dims": [16, 8], "dropout": 0.0, "coeff_reg": 0.01},
    "drn": {"hidden_dims": [16, 8], "dropout": 0.0, "kl_alpha": 0.1},
}

_EXPECTED_TYPES = {
    "cann": CANN,
    "ft_transformer": FTTransformer,
    "tabm": TabMWrapper,
    "cann_gbm": CANNGBM,
    "localglmnet": LocalGLMnet,
    "drn": DistributionalRefinementNetwork,
}


# ---------------------------------------------------------------------------
# Parameterised tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", list(_EXPECTED_TYPES.keys()))
class TestBuildDlModel:
    def test_creates_correct_type(self, arch: str, bundle: DLFeatureBundle) -> None:
        """build_dl_model must return an instance of the expected nn.Module subclass."""
        import torch.nn as nn

        params = _SMALL_PARAMS[arch]
        model = build_dl_model(arch, params, bundle, prediction_floor=1.0)

        assert isinstance(model, nn.Module)
        assert isinstance(model, _EXPECTED_TYPES[arch])

    def test_forward_pass_correct_output_shape(
        self, arch: str, bundle: DLFeatureBundle
    ) -> None:
        """Forward pass must return a (batch,) shaped prediction tensor."""
        params = _SMALL_PARAMS[arch]
        model = build_dl_model(arch, params, bundle, prediction_floor=1.0)
        model.eval()

        x_cont, x_cat, glm_pred = _make_batch()
        with torch.no_grad():
            pred, extra = model(x_cont, x_cat, glm_pred)

        assert pred.shape == torch.Size([_BATCH]), (
            f"Expected shape ({_BATCH},), got {pred.shape}"
        )

    def test_output_is_finite(self, arch: str, bundle: DLFeatureBundle) -> None:
        """All elements of the prediction tensor must be finite (no NaN / Inf)."""
        params = _SMALL_PARAMS[arch]
        model = build_dl_model(arch, params, bundle, prediction_floor=1.0)
        model.eval()

        x_cont, x_cat, glm_pred = _make_batch()
        with torch.no_grad():
            pred, _ = model(x_cont, x_cat, glm_pred)

        assert torch.isfinite(pred).all(), (
            f"Non-finite values in {arch} output: {pred}"
        )

    def test_model_has_trainable_parameters(
        self, arch: str, bundle: DLFeatureBundle
    ) -> None:
        """Every DL architecture must have at least one trainable parameter."""
        params = _SMALL_PARAMS[arch]
        model = build_dl_model(arch, params, bundle, prediction_floor=1.0)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0, f"{arch} has no trainable parameters"


class TestBuildDlModelEdgeCases:
    def test_unknown_architecture_raises_value_error(
        self, bundle: DLFeatureBundle
    ) -> None:
        """build_dl_model must raise ValueError for an unrecognised architecture."""
        with pytest.raises(ValueError, match="Unknown DL architecture"):
            build_dl_model("banana_net", {}, bundle)

    def test_get_default_dl_params_returns_dict(self) -> None:
        """get_default_dl_params must return a non-empty dict for each architecture."""
        for arch in ["cann", "ft_transformer", "tabm", "cann_gbm", "localglmnet", "drn"]:
            params = get_default_dl_params(arch)
            assert isinstance(params, dict)
            assert len(params) > 0

    def test_get_default_dl_params_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_default_dl_params("not_an_arch")


class TestDrnSecondOutput:
    """DRN returns dist_params of shape (batch, 2) as the second output."""

    def test_drn_second_output_shape(self, bundle: DLFeatureBundle) -> None:
        model = build_dl_model("drn", _SMALL_PARAMS["drn"], bundle, prediction_floor=1.0)
        model.eval()

        x_cont, x_cat, glm_pred = _make_batch()
        with torch.no_grad():
            pred, dist_params = model(x_cont, x_cat, glm_pred)

        assert dist_params.shape == torch.Size([_BATCH, 2])

    def test_drn_dist_params_positive(self, bundle: DLFeatureBundle) -> None:
        """Both shape and rate parameters must be strictly positive."""
        model = build_dl_model("drn", _SMALL_PARAMS["drn"], bundle, prediction_floor=1.0)
        model.eval()

        x_cont, x_cat, glm_pred = _make_batch()
        with torch.no_grad():
            _, dist_params = model(x_cont, x_cat, glm_pred)

        assert (dist_params > 0).all(), "DRN dist_params must be strictly positive"
