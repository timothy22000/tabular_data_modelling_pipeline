"""Test that all project modules import without errors or circular imports."""
from __future__ import annotations

import importlib
import sys
import types


def _import(module_name: str) -> types.ModuleType:
    """Import a module by dotted name and return it."""
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Top-level / utility modules
# ---------------------------------------------------------------------------

class TestTopLevelImports:
    def test_dataset_config(self) -> None:
        mod = _import("dataset_config")
        assert hasattr(mod, "DatasetConfig")
        assert hasattr(mod, "auto_detect_features")

    def test_config_example_generic(self) -> None:
        mod = _import("configs.example_generic")
        from dataset_config import DatasetConfig
        assert hasattr(mod, "config")
        assert isinstance(mod.config, DatasetConfig)

    def test_config_net_premium(self) -> None:
        mod = _import("configs.net_premium")
        from dataset_config import DatasetConfig
        assert hasattr(mod, "config")
        assert isinstance(mod.config, DatasetConfig)


# ---------------------------------------------------------------------------
# modelling package
# ---------------------------------------------------------------------------

class TestModellingImports:
    def test_modelling_init(self) -> None:
        _import("modelling")

    def test_modelling_config(self) -> None:
        mod = _import("modelling.config")
        assert hasattr(mod, "DLConfig")
        assert hasattr(mod, "HAS_TORCH")

    def test_modelling_data(self) -> None:
        mod = _import("modelling.data")
        assert hasattr(mod, "DLFeatureBundle")
        assert hasattr(mod, "TabularDataset")
        assert hasattr(mod, "prepare_dl_features")
        assert hasattr(mod, "load_and_prepare_dl_data")
        assert hasattr(mod, "build_dataloaders")

    def test_modelling_losses(self) -> None:
        mod = _import("modelling.losses")
        assert hasattr(mod, "gamma_deviance_loss")
        assert hasattr(mod, "monotonicity_penalty")
        assert hasattr(mod, "drn_loss")

    def test_modelling_training(self) -> None:
        mod = _import("modelling.training")
        assert hasattr(mod, "TrainingResult")
        assert hasattr(mod, "_get_monotone_cont_indices")

    def test_modelling_comparison(self) -> None:
        _import("modelling.comparison")

    def test_modelling_cv(self) -> None:
        _import("modelling.cv")

    def test_modelling_ensemble(self) -> None:
        _import("modelling.ensemble")

    def test_modelling_evaluation(self) -> None:
        _import("modelling.evaluation")

    def test_modelling_interpretability(self) -> None:
        _import("modelling.interpretability")

    def test_modelling_orchestration(self) -> None:
        _import("modelling.orchestration")

    def test_modelling_output(self) -> None:
        _import("modelling.output")

    def test_modelling_pipeline(self) -> None:
        _import("modelling.pipeline")

    def test_modelling_tuning(self) -> None:
        _import("modelling.tuning")

    def test_modelling_visualization(self) -> None:
        _import("modelling.visualization")


# ---------------------------------------------------------------------------
# modelling.models sub-package
# ---------------------------------------------------------------------------

class TestModelSubpackageImports:
    def test_models_init(self) -> None:
        mod = _import("modelling.models")
        assert hasattr(mod, "build_dl_model")
        assert hasattr(mod, "get_default_dl_params")

    def test_models_shared(self) -> None:
        mod = _import("modelling.models.shared")
        assert hasattr(mod, "CategoricalEmbeddings")

    def test_models_cann(self) -> None:
        mod = _import("modelling.models.cann")
        assert hasattr(mod, "CANN")

    def test_models_cann_gbm(self) -> None:
        mod = _import("modelling.models.cann_gbm")
        assert hasattr(mod, "CANNGBM")

    def test_models_ft_transformer(self) -> None:
        mod = _import("modelling.models.ft_transformer")
        assert hasattr(mod, "FTTransformer")

    def test_models_tabm(self) -> None:
        mod = _import("modelling.models.tabm")
        assert hasattr(mod, "TabMWrapper")

    def test_models_localglmnet(self) -> None:
        mod = _import("modelling.models.localglmnet")
        assert hasattr(mod, "LocalGLMnet")

    def test_models_drn(self) -> None:
        mod = _import("modelling.models.drn")
        assert hasattr(mod, "DistributionalRefinementNetwork")

    def test_models_catboost_model(self) -> None:
        _import("modelling.models.catboost_model")

    def test_models_xgboost_model(self) -> None:
        _import("modelling.models.xgboost_model")


# ---------------------------------------------------------------------------
# modelling.utils sub-package
# ---------------------------------------------------------------------------

class TestUtilsImports:
    def test_utils_init(self) -> None:
        _import("modelling.utils")

    def test_utils_metrics(self) -> None:
        mod = _import("modelling.utils.metrics")
        assert hasattr(mod, "compute_gini")
        assert hasattr(mod, "compute_gamma_deviance")
        assert hasattr(mod, "clamp_predictions")

    def test_utils_preprocessing(self) -> None:
        mod = _import("modelling.utils.preprocessing")
        assert hasattr(mod, "cap_target")
        assert hasattr(mod, "load_csv_with_split")

    def test_utils_glm(self) -> None:
        mod = _import("modelling.utils.glm")
        assert hasattr(mod, "fit_gamma_glm")
        assert hasattr(mod, "prepare_design_matrix")
