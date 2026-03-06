"""Loss functions for the DL pipeline."""
from __future__ import annotations

from typing import Any, List, Tuple

from .config import HAS_TORCH, torch, nn, log


if HAS_TORCH:

    def gamma_deviance_loss(
        y_pred: "torch.Tensor", y_true: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute mean Gamma deviance loss for a mini-batch.

        Unit deviance: 2 * [-log(y_true / y_pred) + (y_true - y_pred) / y_pred]
        Both y_pred and y_true are clamped to a minimum of 1.0 to prevent
        numerical instability.

        Args:
            y_pred: Predicted premium values, shape (batch,).
            y_true: Observed premium values, shape (batch,).

        Returns:
            Scalar mean Gamma deviance over the batch.
        """
        y_pred = y_pred.clamp(min=1.0)
        y_true = y_true.clamp(min=1.0)
        ratio = y_true / y_pred
        unit_dev = 2.0 * (-torch.log(ratio) + (y_true - y_pred) / y_pred)
        return unit_dev.mean()

    def monotonicity_penalty(
        model: "nn.Module",
        x_cont: "torch.Tensor",
        x_cat: "torch.Tensor",
        glm_pred: "torch.Tensor",
        feature_indices: List[int],
        directions: List[int],
        epsilon: float = 0.01,
    ) -> "torch.Tensor":
        """Finite-difference monotonicity penalty for constrained DL models.

        For each constrained feature, perturbs the input by +epsilon and
        computes the change in prediction.  If the change violates the
        required direction, the squared violation is accumulated.

        Penalty = sum over constrained features of
            mean( relu( -direction * (f(x + eps*e_k) - f(x)) )^2 )

        This is added to the training loss scaled by ``config.mono_lambda``.

        Args:
            model: The neural network model being trained.  Must accept
                (x_cont, x_cat, glm_pred) and return prediction tensor.
            x_cont: Continuous feature batch, shape (batch, F_cont).
            x_cat: Categorical code batch, shape (batch, F_cat).
            glm_pred: GLM base predictions, shape (batch,).
            feature_indices: Indices into x_cont for constrained features.
            directions: +1 (increasing) or -1 (decreasing) per feature.
            epsilon: Perturbation magnitude (default 0.01 in standardised
                space, approximately 0.01 std of each feature).

        Returns:
            Scalar penalty tensor (0.0 if no constrained features are present
            in the current batch).
        """
        if not feature_indices:
            return torch.tensor(0.0, device=x_cont.device)

        penalty = torch.tensor(0.0, device=x_cont.device)

        with torch.no_grad():
            base_pred, _ = model(x_cont, x_cat, glm_pred)

        for idx, direction in zip(feature_indices, directions):
            x_pert = x_cont.clone()
            x_pert[:, idx] = x_pert[:, idx] + epsilon
            pert_pred, _ = model(x_pert, x_cat, glm_pred)
            delta = pert_pred - base_pred  # positive means increasing
            violation = torch.relu(-direction * delta)
            penalty = penalty + (violation ** 2).mean()

        return penalty / max(len(feature_indices), 1)

    def drn_loss(
        y_true: "torch.Tensor",
        dist_params: "torch.Tensor",
        base_shape: "torch.Tensor",
        glm_pred: "torch.Tensor",
        kl_alpha: float = 0.1,
    ) -> "torch.Tensor":
        """Gamma NLL + KL divergence loss for the Distributional Refinement Network.

        Loss = Gamma NLL(y | shape_refined, rate_refined)
             + kl_alpha * KL(Gamma(shape_refined, rate_refined) || Gamma(base_shape, base_rate))

        The KL divergence acts as a regulariser, penalising excessive departure
        from the GLM base distribution (Penny 2001).

        Args:
            y_true: Observed premium values, shape (batch,).
            dist_params: Stacked [shape_refined, rate_refined], shape (batch, 2).
            base_shape: Scalar base Gamma shape (1/dispersion).
            glm_pred: GLM base predictions, shape (batch,).
            kl_alpha: Weight on the KL regulariser (default 0.1).

        Returns:
            Scalar loss averaged over the batch.
        """
        shape = dist_params[:, 0].clamp(min=1e-6)
        rate = dist_params[:, 1].clamp(min=1e-6)
        y = y_true.clamp(min=1.0)

        # Gamma NLL: -log p(y | shape, rate)
        # = -shape*log(rate) + lgamma(shape) - (shape-1)*log(y) + rate*y
        nll = (
            -shape * torch.log(rate)
            + torch.lgamma(shape)
            - (shape - 1.0) * torch.log(y)
            + rate * y
        )

        # KL(Gamma(shape, rate) || Gamma(shape0, rate0))
        shape0 = base_shape.expand_as(shape).clamp(min=1e-6)
        rate0 = (shape0 / glm_pred.clamp(min=1.0))
        kl = (
            (shape - shape0) * torch.digamma(shape)
            - torch.lgamma(shape)
            + torch.lgamma(shape0)
            + shape0 * (torch.log(rate) - torch.log(rate0))
            + shape * (rate0 - rate) / rate
        )

        loss = nll.mean() + kl_alpha * kl.clamp(min=0.0).mean()
        return loss

else:
    def gamma_deviance_loss(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Stub when PyTorch is unavailable."""
        raise ImportError("PyTorch is required for gamma_deviance_loss.")

    def monotonicity_penalty(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Stub when PyTorch is unavailable."""
        raise ImportError("PyTorch is required for monotonicity_penalty.")

    def drn_loss(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Stub when PyTorch is unavailable."""
        raise ImportError("PyTorch is required for drn_loss.")
