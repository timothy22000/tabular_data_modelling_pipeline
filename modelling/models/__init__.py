"""Model factory and default hyperparameters."""
from __future__ import annotations

from typing import Any, Dict, List

from ..config import HAS_TORCH, nn, log
from ..data import DLFeatureBundle
from .cann import CANN
from .ft_transformer import FTTransformer
from .tabm import TabMWrapper
from .cann_gbm import CANNGBM
from .localglmnet import LocalGLMnet
from .drn import DistributionalRefinementNetwork

ALL_ARCHITECTURES = ["catboost", "xgboost", "cann", "cann_gbm", "ft_transformer", "tabm", "localglmnet", "drn"]


def get_default_dl_params(architecture: str) -> Dict[str, Any]:
    """Return sensible default hyperparameters for a given architecture.

    Defaults are conservative choices validated empirically on UK motor
    datasets of ~25k training rows.

    Args:
        architecture: One of "catboost", "xgboost", "cann", "cann_gbm",
            "ft_transformer", "tabm", "localglmnet", "drn".

    Returns:
        Dictionary of hyperparameter name -> default value.

    Raises:
        ValueError: If ``architecture`` is not a recognised name.
    """
    defaults: Dict[str, Dict[str, Any]] = {
        "catboost": {
            "iterations": 2000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 0.85,
            "bagging_temperature": 1.0,
            "early_stopping_rounds": 50,
        },
        "xgboost": {
            "max_depth": 6,
            "eta": 0.05,
            "n_estimators": 1000,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "early_stopping_rounds": 50,
        },
        "cann": {
            "hidden_dims": [128, 64],
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        },
        "ft_transformer": {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "ffn_factor": 4,
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
        "tabm": {
            "n_members": 8,
            "hidden_dims": [128, 64],
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "cann_gbm": {
            "hidden_dims": [128, 64],
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        },
        "localglmnet": {
            "hidden_dims": [64, 32],
            "dropout": 0.3,
            "coeff_reg": 1.0,
            "lr": 1e-4,
            "weight_decay": 1e-2,
        },
        "drn": {
            "hidden_dims": [128, 64],
            "dropout": 0.2,
            "kl_alpha": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    }

    if architecture not in defaults:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Valid choices: {sorted(defaults.keys())}"
        )
    return dict(defaults[architecture])


def build_dl_model(
    architecture: str,
    params: Dict[str, Any],
    bundle: DLFeatureBundle,
) -> "nn.Module":
    """Factory function: instantiate a DL model from architecture name and params.

    Args:
        architecture: One of "cann", "ft_transformer", "tabm".
        params: Hyperparameter dictionary (may include training params like
            "lr" which are ignored here — only architectural params are used).
        bundle: Feature bundle providing n_cont, category_sizes, embedding_dims.

    Returns:
        Initialised (untrained) nn.Module.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If ``architecture`` is unrecognised.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for build_dl_model().")

    n_cont = bundle.X_train_cont.shape[1]
    category_sizes = [len(m) for m in bundle.category_mappings.values()]
    embedding_dims = bundle.embedding_dims

    if architecture == "cann":
        return CANN(
            n_cont=n_cont,
            category_sizes=category_sizes,
            embedding_dims=embedding_dims,
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout=params.get("dropout", 0.2),
        )

    elif architecture == "ft_transformer":
        return FTTransformer(
            n_cont=n_cont,
            category_sizes=category_sizes,
            d_model=params.get("d_model", 64),
            n_heads=params.get("n_heads", 4),
            n_layers=params.get("n_layers", 3),
            dropout=params.get("dropout", 0.1),
            ffn_factor=params.get("ffn_factor", 4),
        )

    elif architecture == "tabm":
        return TabMWrapper(
            n_cont=n_cont,
            category_sizes=category_sizes,
            embedding_dims=embedding_dims,
            n_members=params.get("n_members", 8),
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout=params.get("dropout", 0.2),
        )

    elif architecture == "cann_gbm":
        return CANNGBM(
            n_cont=n_cont,
            category_sizes=category_sizes,
            embedding_dims=embedding_dims,
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout=params.get("dropout", 0.2),
        )

    elif architecture == "localglmnet":
        return LocalGLMnet(
            n_cont=n_cont,
            category_sizes=category_sizes,
            embedding_dims=embedding_dims,
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout=params.get("dropout", 0.2),
            coeff_reg=params.get("coeff_reg", 0.01),
        )

    elif architecture == "drn":
        return DistributionalRefinementNetwork(
            n_cont=n_cont,
            category_sizes=category_sizes,
            embedding_dims=embedding_dims,
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout=params.get("dropout", 0.2),
            kl_alpha=params.get("kl_alpha", 0.1),
            base_dispersion=getattr(bundle, "glm_dispersion", 1.0),
        )

    else:
        raise ValueError(
            f"Unknown DL architecture: '{architecture}'. "
            "Valid: 'cann', 'cann_gbm', 'ft_transformer', 'tabm', 'localglmnet', 'drn'."
        )
