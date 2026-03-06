"""Training loop, early stopping, and ensemble training for DL models."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    DLConfig,
    HAS_TORCH, torch, nn, DataLoader,
    log, MONOTONE_CONSTRAINTS, _clamp_predictions, _compute_metrics,
)
from .data import DLFeatureBundle, PremiumDataset, build_dataloaders
from .losses import gamma_deviance_loss, monotonicity_penalty


@dataclass
class TrainingResult:
    """Container for results from a single DL model training run.

    Attributes:
        model: The trained nn.Module (on CPU after training completes).
        train_losses: Per-epoch training loss values.
        val_losses: Per-epoch validation loss values.
        best_epoch: Epoch index (0-based) of the best validation checkpoint.
        train_preds: Training set predictions as a float32 NumPy array.
        test_preds: Test set predictions as a float32 NumPy array.
        training_time: Wall-clock seconds for training.
        params: Hyperparameter dict used for this run.
    """

    model: Any  # nn.Module or None
    train_losses: List[float]
    val_losses: List[float]
    best_epoch: int
    train_preds: np.ndarray
    test_preds: np.ndarray
    training_time: float
    params: Dict[str, Any]


if HAS_TORCH:

    class EarlyStopping:
        """Tracks validation loss and saves the best model state dict.

        Triggers when validation loss has not improved for ``patience``
        consecutive epochs.  The best model state is restored when
        ``should_stop`` returns True.

        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum absolute improvement to count as an improvement.
        """

        def __init__(self, patience: int = 30, min_delta: float = 1e-6) -> None:
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss: float = float("inf")
            self.best_state: Optional[Dict[str, Any]] = None
            self.counter: int = 0
            self.best_epoch: int = 0

        def step(
            self,
            val_loss: float,
            model: nn.Module,
            epoch: int,
        ) -> bool:
            """Update state and determine whether training should stop.

            Args:
                val_loss: Current epoch's validation loss.
                model: Current model (state dict saved if improved).
                epoch: Current epoch index (0-based).

            Returns:
                True when patience is exhausted and training should stop.
            """
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.best_state = {
                    k: v.clone().cpu() for k, v in model.state_dict().items()
                }
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
            return self.counter >= self.patience

        def restore_best(self, model: nn.Module) -> None:
            """Load the best saved state dict back into the model.

            Args:
                model: Model instance to restore the best weights into.
            """
            if self.best_state is not None:
                model.load_state_dict(self.best_state)

else:

    class EarlyStopping:  # type: ignore[no-redef]
        """Stub EarlyStopping when PyTorch is unavailable."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for EarlyStopping.")


def _resolve_device(config: DLConfig) -> "torch.device":
    """Resolve the target compute device from config.device.

    Priority: MPS (Apple Silicon) > CUDA > CPU.

    Args:
        config: DL pipeline configuration with ``device`` field.

    Returns:
        A ``torch.device`` instance.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to resolve a compute device.")

    if config.device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("  Device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            log.info("  Device: CUDA (%s)", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            log.info("  Device: CPU")
    else:
        device = torch.device(config.device)
        log.info("  Device: %s (user specified)", config.device)

    return device


def _get_monotone_cont_indices(
    continuous_feature_names: List[str],
) -> Tuple[List[int], List[int]]:
    """Get indices and directions for monotonicity-constrained continuous features.

    Only continuous features are penalised (categorical effects cannot be
    simply perturbed along a single dimension).

    Args:
        continuous_feature_names: Ordered list of continuous feature names.

    Returns:
        Tuple of (feature_indices, directions) for features present in
        MONOTONE_CONSTRAINTS with non-zero constraint.
    """
    indices: List[int] = []
    directions: List[int] = []
    for i, name in enumerate(continuous_feature_names):
        direction = MONOTONE_CONSTRAINTS.get(name, 0)
        if direction != 0:
            indices.append(i)
            directions.append(direction)
    return indices, directions


def train_one_epoch(
    model: "nn.Module",
    loader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    device: "torch.device",
    config: DLConfig,
    mono_indices: List[int],
    mono_directions: List[int],
    base_mode: str = "none",
) -> float:
    """Train for one epoch, returning the mean batch loss.

    Each mini-batch loss is the sum of:
      - Gamma deviance loss (or DRN loss when base_mode=="drn") on the batch
        predictions.
      - ``config.mono_lambda`` * monotonicity penalty (finite difference).

    Gradients are clipped to max_norm=1.0 before the optimiser step.

    Args:
        model: The neural network being trained.
        loader: DataLoader yielding (x_cont, x_cat, glm_pred, gbm_pred, y)
            batches.
        optimizer: AdamW optimiser.
        device: Target compute device.
        config: DL pipeline configuration (mono_lambda, batch_size).
        mono_indices: Indices of constrained continuous features.
        mono_directions: Corresponding monotone direction (+1/-1).
        base_mode: Controls which base prediction is passed to the model and
            which loss function is applied.  One of ``"glm"``, ``"gbm"``,
            ``"drn"``, or ``"none"``.

    Returns:
        Mean loss averaged over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_cont, x_cat, glm_pred, gbm_pred, y in loader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        glm_pred = glm_pred.to(device)
        gbm_pred = gbm_pred.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Resolve base prediction for this architecture
        if base_mode == "gbm":
            base_pred = gbm_pred
        elif base_mode in ("glm", "drn"):
            base_pred = glm_pred
        else:
            base_pred = glm_pred  # default for CANN-like models

        pred, extra = model(x_cont, x_cat, base_pred)

        if base_mode == "drn":
            from .losses import drn_loss
            loss = drn_loss(y, extra, model.base_shape, glm_pred, model.kl_alpha)
        else:
            loss = gamma_deviance_loss(pred, y)

        # LocalGLMnet coefficient L2 regularisation
        if hasattr(model, "coeff_reg") and model.coeff_reg > 0:
            loss = loss + model.coeff_reg * extra.pow(2).mean()

        # Monotonicity penalty
        if config.mono_lambda > 0 and mono_indices:
            penalty = monotonicity_penalty(
                model, x_cont, x_cat, base_pred,
                mono_indices, mono_directions,
            )
            loss = loss + config.mono_lambda * penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: "nn.Module",
    loader: "DataLoader",
    device: "torch.device",
    base_mode: str = "none",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader, returning loss and predictions.

    Args:
        model: Neural network in eval mode.
        loader: DataLoader yielding (x_cont, x_cat, glm_pred, gbm_pred, y)
            batches.
        device: Compute device.
        base_mode: Controls which base prediction is passed to the model and
            which loss function is applied.  One of ``"glm"``, ``"gbm"``,
            ``"drn"``, or ``"none"``.

    Returns:
        Tuple of:
            mean_loss: Mean loss (Gamma deviance or DRN loss) over the full
                loader.
            all_preds: Concatenated predictions as float32 NumPy array.
            all_actuals: Concatenated actuals as float32 NumPy array.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: List[np.ndarray] = []
    all_actuals: List[np.ndarray] = []

    with torch.no_grad():
        for x_cont, x_cat, glm_pred, gbm_pred, y in loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            glm_pred = glm_pred.to(device)
            gbm_pred = gbm_pred.to(device)
            y = y.to(device)

            if base_mode == "gbm":
                base_pred = gbm_pred
            elif base_mode in ("glm", "drn"):
                base_pred = glm_pred
            else:
                base_pred = glm_pred

            pred, extra = model(x_cont, x_cat, base_pred)

            if base_mode == "drn":
                from .losses import drn_loss
                loss = drn_loss(y, extra, model.base_shape, glm_pred, model.kl_alpha)
            else:
                loss = gamma_deviance_loss(pred, y)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.cpu().numpy())

    mean_loss = total_loss / max(n_batches, 1)
    preds_arr = np.concatenate(all_preds).astype(np.float32)
    actuals_arr = np.concatenate(all_actuals).astype(np.float32)

    return mean_loss, preds_arr, actuals_arr


def train_dl_model(
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    test_loader: "DataLoader",
    config: DLConfig,
    arch_name: str,
    continuous_feature_names: List[str],
    base_mode: str = "none",
    full_train_loader: Any = None,
    params: Optional[Dict[str, Any]] = None,
) -> TrainingResult:
    """Full training loop for a single DL architecture.

    Training procedure:
      1. Move model to device.
      2. For CANN / CANN-GBM: freeze all NN parameters for the first 10 epochs
         so the base model can be evaluated before the residual network adapts.
      3. AdamW optimiser with CosineAnnealingWarmRestarts scheduler.
      4. Early stopping monitors validation loss.
      5. Restore best checkpoint; collect train and test predictions.

    Args:
        model: Initialised nn.Module ready for training.
        train_loader: DataLoader for training mini-batches.
        val_loader: DataLoader for validation (loss tracking only).
        test_loader: DataLoader for final test predictions.
        config: DL pipeline configuration.
        arch_name: Architecture label for logging (e.g. "cann").
        continuous_feature_names: Names aligned to x_cont columns (for
            monotonicity constraint lookup).
        base_mode: Controls which base prediction is used and which loss
            function is applied.  One of ``"glm"``, ``"gbm"``, ``"drn"``,
            or ``"none"``.  CANN-type architectures (``"glm"`` and ``"gbm"``)
            receive a 10-epoch parameter freeze phase for the base warm-up.

    Returns:
        TrainingResult dataclass with model, predictions, and loss curves.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for train_dl_model().")

    device = _resolve_device(config)
    model = model.to(device)

    mono_indices, mono_directions = _get_monotone_cont_indices(continuous_feature_names)

    epochs = config.epochs
    if config.quick:
        epochs = min(epochs, 50)

    # CANN / CANN-GBM: freeze MLP parameters for first 10 warm-up epochs.
    # Only applies to models with a .mlp attribute (CANN, CANNGBM) — not
    # LocalGLMnet (which has .coeff_net) or DRN (which has .trunk).
    if params is None:
        params = {}
    lr = params.get("lr", 1e-3)
    weight_decay = params.get("weight_decay", 1e-4)

    is_cann_type = base_mode in ("glm", "gbm") and hasattr(model, "mlp")
    cann_freeze_epochs = 3 if is_cann_type else 0
    if is_cann_type and cann_freeze_epochs > 0:
        log.info("  [%s] Freezing NN for first %d epochs (base warm-up)", arch_name, cann_freeze_epochs)
        for param in model.mlp.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    early_stop = EarlyStopping(patience=config.patience, min_delta=1e-6)

    train_losses: List[float] = []
    val_losses: List[float] = []
    t0 = time.time()

    for epoch in range(epochs):
        # Unfreeze CANN / CANN-GBM MLP after warm-up phase
        if is_cann_type and epoch == cann_freeze_epochs:
            log.info("  [%s] Unfreezing NN at epoch %d", arch_name, epoch)
            for param in model.mlp.parameters():
                param.requires_grad = True
            # Re-create optimiser to include newly unfrozen parameters
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2
            )

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, config,
            mono_indices, mono_directions, base_mode,
        )
        val_loss, _, _ = evaluate(model, val_loader, device, base_mode)

        scheduler.step(epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(
                "  [%s] Epoch %3d/%d — train_loss=%.6f  val_loss=%.6f  lr=%.2e",
                arch_name,
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                optimizer.param_groups[0]["lr"],
            )

        if early_stop.step(val_loss, model, epoch):
            log.info(
                "  [%s] Early stopping at epoch %d (best epoch=%d, best_val=%.6f)",
                arch_name,
                epoch + 1,
                early_stop.best_epoch + 1,
                early_stop.best_loss,
            )
            break

    early_stop.restore_best(model)
    model = model.cpu()
    elapsed = time.time() - t0

    # Collect predictions on FULL training set and test set
    model = model.to(device)
    pred_train_loader = full_train_loader if full_train_loader is not None else train_loader
    _, train_preds_arr, _ = evaluate(model, pred_train_loader, device, base_mode)
    _, test_preds_arr, _ = evaluate(model, test_loader, device, base_mode)
    model = model.cpu()

    train_preds_arr = _clamp_predictions(train_preds_arr)
    test_preds_arr = _clamp_predictions(test_preds_arr)

    log.info(
        "  [%s] Training complete — %.1fs | best_epoch=%d | best_val=%.6f",
        arch_name,
        elapsed,
        early_stop.best_epoch + 1,
        early_stop.best_loss,
    )

    return TrainingResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=early_stop.best_epoch,
        train_preds=train_preds_arr,
        test_preds=test_preds_arr,
        training_time=elapsed,
        params={},
    )


def _train_dl_ensemble(
    architecture: str,
    params: Dict[str, Any],
    bundle: DLFeatureBundle,
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    test_loader: "DataLoader",
    config: DLConfig,
) -> Dict[str, Any]:
    """Train n_ensemble DL models with different seeds and average predictions.

    For each seed in [config.seed, config.seed+1, ..., config.seed+n_ensemble-1]:
      - Build a fresh model instance.
      - Set Python / NumPy / PyTorch seeds for reproducibility.
      - Train with train_dl_model.
      - Collect train_preds and test_preds.

    Final predictions are the arithmetic mean across all ensemble members.

    Args:
        architecture: Architecture identifier passed to build_dl_model.
        params: Best hyperparameter dict from tuning.
        bundle: Feature bundle.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        test_loader: DataLoader for test.
        config: DL pipeline configuration (n_ensemble, seed).

    Returns:
        Dictionary with keys:
            - "ensemble_results": List of TrainingResult per member.
            - "train_preds": Averaged training predictions (float32 array).
            - "test_preds": Averaged test predictions (float32 array).
            - "metrics_train": Metric dict on averaged train predictions.
            - "metrics_test": Metric dict on averaged test predictions.
            - "training_time": Total wall-clock seconds across all members.
    """
    from .models import build_dl_model

    n_ensemble = config.n_ensemble
    if config.quick:
        n_ensemble = 1
        log.info("  [quick] n_ensemble reduced to 1")

    ensemble_results: List[TrainingResult] = []
    all_train_preds: List[np.ndarray] = []
    all_test_preds: List[np.ndarray] = []
    total_time = 0.0

    base_mode_map = {
        "cann": "glm",
        "cann_gbm": "gbm",
        "localglmnet": "glm",
        "drn": "drn",
    }
    base_mode = base_mode_map.get(architecture, "none")

    # Build a full-training DataLoader (no val split) for final predictions
    full_train_dataset = PremiumDataset(
        x_cont=bundle.X_train_cont,
        x_cat=bundle.X_train_cat,
        y=bundle.y_train,
        glm_preds=bundle.glm_train_preds,
        gbm_preds=bundle.gbm_train_preds,
    )
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    for member_idx in range(n_ensemble):
        member_seed = config.seed + member_idx
        log.info(
            "  [%s] Ensemble member %d/%d (seed=%d)",
            architecture,
            member_idx + 1,
            n_ensemble,
            member_seed,
        )

        # Set seeds for this member
        np.random.seed(member_seed)
        if HAS_TORCH:
            torch.manual_seed(member_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(member_seed)

        model = build_dl_model(architecture, params, bundle)
        result = train_dl_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            arch_name=f"{architecture}_m{member_idx}",
            continuous_feature_names=bundle.continuous_feature_names,
            base_mode=base_mode,
            full_train_loader=full_train_loader,
            params=params,
        )
        result.params = params

        ensemble_results.append(result)
        all_train_preds.append(result.train_preds)
        all_test_preds.append(result.test_preds)
        total_time += result.training_time

    # Average across ensemble members
    avg_train_preds = _clamp_predictions(
        np.stack(all_train_preds, axis=0).mean(axis=0)
    )
    avg_test_preds = _clamp_predictions(
        np.stack(all_test_preds, axis=0).mean(axis=0)
    )

    metrics_train = _compute_metrics(bundle.y_train, avg_train_preds, "train")
    metrics_test = _compute_metrics(bundle.y_test, avg_test_preds, "test")

    log.info(
        "  [%s] Ensemble (%d members) — Train Gini: %.4f | Test Gini: %.4f | "
        "Test MAE: %.0f | A/E: %.4f",
        architecture,
        n_ensemble,
        metrics_train["gini"],
        metrics_test["gini"],
        metrics_test["mae"],
        metrics_test["ae_ratio"],
    )

    return {
        "ensemble_results": ensemble_results,
        "train_preds": avg_train_preds,
        "test_preds": avg_test_preds,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "training_time": total_time,
    }
