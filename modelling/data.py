"""Data loading and feature preparation for the tabular modelling pipeline.

This module provides data ingestion, feature engineering, and PyTorch Dataset /
DataLoader construction used by all DL and GBM benchmark architectures.
"""

from .config import (
    DLConfig,
    HAS_TORCH, HAS_CATBOOST,
    torch, nn, DataLoader, Dataset, random_split,
    Pool,
    np, pd, log,
    DatasetConfig,
    prepare_design_matrix, align_test_matrix, fit_gamma_glm,
    compute_gini, _clamp_predictions,
)
from .utils.preprocessing import load_csv_with_split, cap_target
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Section 1: Data Loading
# =============================================================================


def load_and_prepare_dl_data(
    config: DLConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load CSV, split train/test, cap target, apply optional consolidation.

    Args:
        config: DL pipeline configuration (includes dataset config).

    Returns:
        Tuple of (train_df, test_df, cap_value).
    """
    log.info("=" * 72)
    log.info("SECTION 1: Data Loading")
    log.info("=" * 72)

    ds = config.dataset

    # Load and split
    train_df, test_df = load_csv_with_split(
        input_path=config.input_path,
        target_col=ds.target_col,
        split_col=ds.split_col,
        exclude_cols=ds.exclude_cols,
        seed=config.seed,
        quick=config.quick,
    )

    # Apply categorical consolidation if provided
    if ds.categorical_consolidation is not None:
        train_df = ds.categorical_consolidation(train_df)
        test_df = ds.categorical_consolidation(test_df)
        log.info("  Applied categorical consolidation")

    # Cap target on training data
    train_df, cap_value = cap_target(
        train_df,
        ds.target_col,
        cap_percentile=ds.cap_percentile,
        cap_value=ds.cap_value,
    )

    # Also cap test for consistent column structure
    capped_col = f"{ds.target_col}_CAPPED"
    test_df[capped_col] = test_df[ds.target_col].clip(upper=cap_value)

    # Compute derived features
    for feat_name, feat_fn in ds.derived_features.items():
        try:
            train_df[feat_name] = feat_fn(train_df)
            test_df[feat_name] = feat_fn(test_df)
        except Exception as exc:
            log.warning("  Derived feature '%s' failed: %s", feat_name, exc)

    log.info(
        "  Data ready — Train: %d rows | Test: %d rows | Cap: %.2f",
        len(train_df),
        len(test_df),
        cap_value,
    )
    return train_df, test_df, cap_value


# =============================================================================
# Section 2: Feature Preparation
# =============================================================================


@dataclass
class DLFeatureBundle:
    """All feature matrices and metadata required by DL and GBM architectures.

    Attributes:
        X_train_cont: Standardised continuous feature matrix, shape (N_train, F_cont).
        X_test_cont: Standardised continuous feature matrix, shape (N_test, F_cont).
        X_train_cat: Integer-encoded categorical matrix, shape (N_train, F_cat).
            0 is the UNKNOWN token; category codes start at 1.
        X_test_cat: Integer-encoded categorical matrix, shape (N_test, F_cat).
        y_train: Capped training response (positive, clipped to floor=1).
        y_test: Uncapped test response (honest evaluation target).
        continuous_feature_names: Ordered names of continuous features.
        categorical_feature_names: Ordered names of categorical features.
        category_mappings: Mapping of {col: {raw_str: int_code}} for each
            categorical, with UNKNOWN -> 0 sentinel.
        embedding_dims: Suggested embedding dimension per categorical column,
            computed as min(50, (n_categories + 1) // 2).
        cont_mean: Per-feature training mean used for standardisation.
        cont_std: Per-feature training std used for standardisation (floored
            at 1e-8 to avoid division by zero).
        glm_train_preds: GLM base predictions on training set (for CANN).
        glm_test_preds: GLM base predictions on test set (for CANN).
        catboost_train_pool: CatBoost Pool object for training (None if
            CatBoost is not installed or pool construction fails).
        catboost_test_pool: CatBoost Pool object for test.
        train_df: Raw consolidated training DataFrame (for rebuilding
            design matrices if needed).
        test_df: Raw consolidated test DataFrame.
        gbm_train_preds: CatBoost predictions on train set (for CANN-GBM).
        gbm_test_preds: CatBoost predictions on test set (for CANN-GBM).
        glm_dispersion: GLM dispersion parameter (for DRN).
    """

    X_train_cont: np.ndarray
    X_test_cont: np.ndarray
    X_train_cat: np.ndarray
    X_test_cat: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    continuous_feature_names: List[str]
    categorical_feature_names: List[str]
    category_mappings: Dict[str, Dict[str, int]]
    embedding_dims: List[int]
    cont_mean: np.ndarray
    cont_std: np.ndarray
    glm_train_preds: np.ndarray
    glm_test_preds: np.ndarray
    catboost_train_pool: Any  # catboost.Pool or None
    catboost_test_pool: Any  # catboost.Pool or None
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    gbm_train_preds: Optional[np.ndarray] = None   # CatBoost predictions on train set (for CANN-GBM)
    gbm_test_preds: Optional[np.ndarray] = None     # CatBoost predictions on test set (for CANN-GBM)
    glm_dispersion: float = 1.0                     # GLM dispersion parameter (for DRN)


def _build_category_mappings(
    train_series: pd.Series,
    test_series: pd.Series,
) -> Tuple[Dict[str, int], pd.Series, pd.Series]:
    """Build integer encoding for a categorical column.

    Unseen test categories (including nulls) are mapped to the UNKNOWN token
    (code = 0).  Training categories are enumerated starting from 1.

    Args:
        train_series: Training categorical column (string dtype).
        test_series: Test categorical column (string dtype).

    Returns:
        Tuple of:
            mapping: {raw_string: int_code} with UNKNOWN -> 0.
            encoded_train: Integer-coded training Series.
            encoded_test: Integer-coded test Series (unseen -> 0).
    """
    unique_cats = sorted(train_series.dropna().unique().tolist())
    mapping: Dict[str, int] = {"UNKNOWN": 0}
    for i, cat in enumerate(unique_cats, start=1):
        mapping[cat] = i

    encoded_train = train_series.map(mapping).fillna(0).astype(int)
    encoded_test = test_series.map(mapping).fillna(0).astype(int)

    return mapping, encoded_train, encoded_test


def _compute_embedding_dim(n_categories: int) -> int:
    """Compute suggested embedding dimensionality for a categorical variable.

    Uses the common heuristic: min(50, (n_categories + 1) // 2), which
    balances expressiveness with parameter economy.

    Args:
        n_categories: Number of distinct training categories (excluding UNKNOWN).

    Returns:
        Suggested embedding dimension (at least 1).
    """
    return max(1, min(50, (n_categories + 1) // 2))


def _build_glm_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    config: DLConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit a Gamma GLM and return training predictions, test predictions, and dispersion.

    Uses the GLM factors from DatasetConfig to build the design matrix.
    """
    ds = config.dataset
    floor = ds.prediction_floor

    log.info("  Building GLM design matrix for CANN base predictions ...")
    y_train_series = pd.Series(y_train, index=train_df.index)

    # Determine which factors are categorical
    cat_set = set(ds.categorical_features)

    X_train_glm = prepare_design_matrix(
        train_df, ds.glm_factors, ds.base_levels, categorical_factors=cat_set
    )
    X_test_glm = prepare_design_matrix(
        test_df, ds.glm_factors, ds.base_levels, categorical_factors=cat_set
    )
    X_test_glm = align_test_matrix(X_train_glm, X_test_glm)

    glm_result = fit_gamma_glm(X_train_glm, y_train_series)

    glm_train_preds = _clamp_predictions(
        np.asarray(glm_result.predict(X_train_glm), dtype=np.float32), floor
    )
    glm_test_preds = _clamp_predictions(
        np.asarray(glm_result.predict(X_test_glm), dtype=np.float32), floor
    )

    try:
        glm_dispersion = float(glm_result.scale)
    except AttributeError:
        glm_dispersion = 1.0
        log.info("  GLM dispersion not available — using 1.0")

    # Guard against NaN / non-positive dispersion (can happen when the
    # GLM fit hits a degenerate residual structure). DRN consumes this
    # as a fixed scale parameter; a NaN here poisons every prediction.
    if not (glm_dispersion > 0 and np.isfinite(glm_dispersion)):
        log.warning(
            "  GLM dispersion = %s is invalid — falling back to 1.0",
            glm_dispersion,
        )
        glm_dispersion = 1.0

    glm_gini_train = compute_gini(y_train, glm_train_preds)
    log.info(
        "  GLM base — Train Gini: %.4f  |  n_params: %d  |  dispersion: %.6f",
        glm_gini_train,
        glm_result.n_params,
        glm_dispersion,
    )

    return glm_train_preds, glm_test_preds, glm_dispersion


def _build_catboost_pools(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cat_feature_indices: List[int],
) -> Tuple[Any, Any]:
    """Construct CatBoost Pool objects for train and test data.

    CatBoost pools are built on the RAW (unstandardised) feature matrices,
    as CatBoost handles its own internal normalisation.

    Args:
        X_train_raw: Raw (non-standardised) training feature DataFrame.
        X_test_raw: Raw (non-standardised) test feature DataFrame.
        y_train: Training response values.
        y_test: Test response values.
        cat_feature_indices: Column indices of categorical features.

    Returns:
        Tuple of (train_pool, test_pool).  Returns (None, None) if CatBoost
        is not installed or pool construction raises an exception.
    """
    if not HAS_CATBOOST:
        log.warning("  CatBoost not installed — skipping Pool construction")
        return None, None

    try:
        train_pool = Pool(
            data=X_train_raw,
            label=y_train,
            cat_features=cat_feature_indices,
        )
        test_pool = Pool(
            data=X_test_raw,
            label=y_test,
            cat_features=cat_feature_indices,
        )
        log.info(
            "  CatBoost pools built — Train: %d rows | Test: %d rows",
            len(y_train),
            len(y_test),
        )
        return train_pool, test_pool
    except Exception as exc:
        log.warning("  CatBoost Pool construction failed: %s", exc)
        return None, None


def prepare_dl_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: DLConfig,
) -> DLFeatureBundle:
    """Build all feature representations required by the DL and GBM architectures.

    Execution order:
      1. Call ``prepare_gbm_features`` to derive continuous + categorical matrices.
      2. Build per-category integer encodings with UNKNOWN=0 sentinel.
      3. Compute embedding dimensions for each categorical column.
      4. Z-score standardise continuous features using training statistics only.
      5. Pre-compute GLM base predictions for CANN.
      6. Construct CatBoost Pool objects for GBM architectures.
      7. Assemble and return a DLFeatureBundle.

    Args:
        train_df: Consolidated training DataFrame (from load_and_prepare_dl_data).
        test_df: Consolidated test DataFrame.
        config: DL pipeline configuration.

    Returns:
        DLFeatureBundle containing all feature matrices, metadata, GLM
        predictions, and CatBoost pools.
    """
    log.info("=" * 72)
    log.info("SECTION 2: Feature Preparation")
    log.info("=" * 72)

    ds = config.dataset
    target_col = ds.target_col
    capped_col = f"{target_col}_CAPPED"

    # ----- Step 1: Build feature matrices from DatasetConfig ------------------
    # Extract target
    y_train = train_df[capped_col].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

    # Clamp to floor
    y_train = np.maximum(y_train, ds.prediction_floor).astype(np.float32)
    y_test = np.maximum(y_test, ds.prediction_floor).astype(np.float32)

    # Get feature names from config (with derived features included)
    all_continuous_names = list(ds.continuous_features) + list(ds.derived_features.keys())
    cont_names_present = [n for n in all_continuous_names if n in train_df.columns]
    cat_names_present = [n for n in ds.categorical_features if n in train_df.columns]

    log.info(
        "  Feature split — %d continuous, %d categorical",
        len(cont_names_present),
        len(cat_names_present),
    )

    # Raw (non-standardised) continuous arrays for CatBoost and XGBoost
    X_train_cont_raw = train_df[cont_names_present].fillna(-999.0).values.astype(np.float32)
    X_test_cont_raw = test_df[cont_names_present].fillna(-999.0).values.astype(np.float32)

    # Raw combined DataFrame for CatBoost (needs string categoricals)
    X_train_raw_df = train_df[cont_names_present + cat_names_present].copy()
    X_test_raw_df = test_df[cont_names_present + cat_names_present].copy()

    # Ensure categoricals are strings for CatBoost Pool
    for col in cat_names_present:
        X_train_raw_df[col] = train_df[col].astype(str).fillna("UNKNOWN")
        X_test_raw_df[col] = test_df[col].astype(str).fillna("UNKNOWN")

    # ----- Step 2: Build per-category integer encodings ----------------------
    category_mappings: Dict[str, Dict[str, int]] = {}
    cat_encoded_train_cols: List[np.ndarray] = []
    cat_encoded_test_cols: List[np.ndarray] = []

    for col in cat_names_present:
        raw_tr = train_df[col].astype(str).fillna("UNKNOWN")
        raw_te = test_df[col].astype(str).fillna("UNKNOWN")
        mapping, enc_tr, enc_te = _build_category_mappings(raw_tr, raw_te)
        category_mappings[col] = mapping
        cat_encoded_train_cols.append(enc_tr.values.astype(np.int64))
        cat_encoded_test_cols.append(enc_te.values.astype(np.int64))
        n_cats = len(mapping) - 1  # exclude UNKNOWN sentinel
        log.info("    %s: %d categories (embedding_dim=%d)", col, n_cats, _compute_embedding_dim(n_cats))

    X_train_cat = (
        np.stack(cat_encoded_train_cols, axis=1)
        if cat_encoded_train_cols
        else np.zeros((len(y_train), 0), dtype=np.int64)
    )
    X_test_cat = (
        np.stack(cat_encoded_test_cols, axis=1)
        if cat_encoded_test_cols
        else np.zeros((len(y_test), 0), dtype=np.int64)
    )

    # ----- Step 3: Compute embedding dimensions ------------------------------
    embedding_dims: List[int] = []
    for col in cat_names_present:
        n_cats = len(category_mappings[col]) - 1
        embedding_dims.append(_compute_embedding_dim(n_cats))

    log.info("  Embedding dims per categorical: %s", dict(zip(cat_names_present, embedding_dims)))

    # ----- Step 4: Z-score standardise continuous features -------------------
    cont_mean = np.nanmean(X_train_cont_raw, axis=0).astype(np.float32)
    cont_std = np.nanstd(X_train_cont_raw, axis=0).astype(np.float32)
    cont_std = np.where(cont_std < 1e-8, 1.0, cont_std)  # avoid zero division

    X_train_cont = ((X_train_cont_raw - cont_mean) / cont_std).astype(np.float32)
    X_test_cont = ((X_test_cont_raw - cont_mean) / cont_std).astype(np.float32)

    # Replace any residual NaN from standardisation with 0 (standardised mean)
    X_train_cont = np.nan_to_num(X_train_cont, nan=0.0)
    X_test_cont = np.nan_to_num(X_test_cont, nan=0.0)

    log.info(
        "  Continuous features standardised — mean in [%.3f, %.3f], std in [%.3f, %.3f]",
        float(cont_mean.min()),
        float(cont_mean.max()),
        float(cont_std.min()),
        float(cont_std.max()),
    )

    # ----- Step 5: Pre-compute GLM base predictions for CANN -----------------
    needs_glm = {"cann", "cann_gbm", "localglmnet", "drn"}.intersection(config.architectures)
    if needs_glm:
        glm_train_preds, glm_test_preds, glm_dispersion = _build_glm_predictions(
            train_df, test_df, y_train, config
        )
    else:
        glm_train_preds = np.ones(len(y_train), dtype=np.float32)
        glm_test_preds = np.ones(len(y_test), dtype=np.float32)
        glm_dispersion = 1.0
        log.info("  No GLM-based architectures — GLM predictions set to 1.0 (dummy)")

    # ----- Step 6: Build CatBoost Pool objects -------------------------------
    # CatBoost requires the raw string DataFrame with cat_features as column indices.
    cat_col_indices_in_df = [
        list(X_train_raw_df.columns).index(c)
        for c in cat_names_present
        if c in X_train_raw_df.columns
    ]

    catboost_train_pool, catboost_test_pool = _build_catboost_pools(
        X_train_raw_df,
        X_test_raw_df,
        y_train,
        y_test,
        cat_col_indices_in_df,
    )

    # ----- Step 7: Assemble and return bundle --------------------------------
    bundle = DLFeatureBundle(
        X_train_cont=X_train_cont,
        X_test_cont=X_test_cont,
        X_train_cat=X_train_cat,
        X_test_cat=X_test_cat,
        y_train=y_train,
        y_test=y_test,
        continuous_feature_names=cont_names_present,
        categorical_feature_names=cat_names_present,
        category_mappings=category_mappings,
        embedding_dims=embedding_dims,
        cont_mean=cont_mean,
        cont_std=cont_std,
        glm_train_preds=glm_train_preds,
        glm_test_preds=glm_test_preds,
        catboost_train_pool=catboost_train_pool,
        catboost_test_pool=catboost_test_pool,
        train_df=train_df,
        test_df=test_df,
        glm_dispersion=glm_dispersion,
    )

    log.info(
        "  DLFeatureBundle ready — cont=%d, cat=%d, train=%d, test=%d",
        X_train_cont.shape[1],
        X_train_cat.shape[1],
        len(y_train),
        len(y_test),
    )
    return bundle


# ---------------------------------------------------------------------------
# 2a: PyTorch Dataset and DataLoader builders
# ---------------------------------------------------------------------------


if HAS_TORCH:

    class TabularDataset(Dataset):  # type: ignore[misc]
        """PyTorch Dataset wrapping continuous features, categorical codes,
        optional GLM base predictions, optional GBM predictions, and the target.

        Args:
            x_cont: Float32 array of standardised continuous features,
                shape (N, F_cont).
            x_cat: Int64 array of categorical integer codes, shape (N, F_cat).
            y: Float32 target array, shape (N,).
            glm_preds: Float32 GLM base predictions, shape (N,).
                Defaults to a ones array when CANN is not used.
            gbm_preds: Float32 CatBoost base predictions, shape (N,).
                Defaults to a ones array when CANN-GBM is not used.
        """

        def __init__(
            self,
            x_cont: np.ndarray,
            x_cat: np.ndarray,
            y: np.ndarray,
            glm_preds: Optional[np.ndarray] = None,
            gbm_preds: Optional[np.ndarray] = None,
        ) -> None:
            self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
            self.x_cat = torch.tensor(x_cat, dtype=torch.long)
            self.y = torch.tensor(y, dtype=torch.float32)
            if glm_preds is None:
                glm_preds = np.ones(len(y), dtype=np.float32)
            self.glm_preds = torch.tensor(glm_preds, dtype=torch.float32)
            if gbm_preds is None:
                gbm_preds = np.ones(len(y), dtype=np.float32)
            self.gbm_preds = torch.tensor(gbm_preds, dtype=torch.float32)

        def __len__(self) -> int:
            """Return number of observations in the dataset."""
            return len(self.y)

        def __getitem__(
            self, idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Return (x_cont, x_cat, glm_pred, gbm_pred, y) for a single observation.

            Args:
                idx: Integer index into the dataset.

            Returns:
                Tuple of (continuous_features, categorical_codes,
                    glm_prediction, gbm_prediction, target) tensors.
            """
            return (
                self.x_cont[idx],
                self.x_cat[idx],
                self.glm_preds[idx],
                self.gbm_preds[idx],
                self.y[idx],
            )

else:
    # Provide a stub so module-level references don't raise NameError
    class TabularDataset:  # type: ignore[no-redef]
        """Stub TabularDataset when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for TabularDataset.")


def build_dataloaders(
    bundle: DLFeatureBundle,
    config: DLConfig,
) -> Tuple[Any, Any, Any]:
    """Build train, validation, and test DataLoaders.

    Splits the training set into a train / validation partition using
    ``config.val_fraction``.  The test DataLoader is built from the full test
    set with ``shuffle=False``.

    Args:
        bundle: Feature bundle containing all arrays.
        config: DL pipeline configuration.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).  Returns
        (None, None, None) when PyTorch is unavailable.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not HAS_TORCH:
        log.warning("  PyTorch not installed — cannot build DataLoaders")
        return None, None, None

    rng = torch.Generator().manual_seed(config.seed)

    full_train_dataset = TabularDataset(
        x_cont=bundle.X_train_cont,
        x_cat=bundle.X_train_cat,
        y=bundle.y_train,
        glm_preds=bundle.glm_train_preds,
        gbm_preds=bundle.gbm_train_preds,  # NEW
    )

    n_total = len(full_train_dataset)
    n_val = max(1, int(n_total * config.val_fraction))
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_train_dataset, [n_train, n_val], generator=rng
    )

    test_dataset = TabularDataset(
        x_cont=bundle.X_test_cont,
        x_cat=bundle.X_test_cat,
        y=bundle.y_test,
        glm_preds=bundle.glm_test_preds,
        gbm_preds=bundle.gbm_test_preds,  # NEW
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    log.info(
        "  DataLoaders — train: %d batches | val: %d batches | test: %d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader
