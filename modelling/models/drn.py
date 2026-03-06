"""DRN — Distributional Refinement Network (Avanzi et al. 2023)."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..config import HAS_TORCH, torch, nn
from .shared import CategoricalEmbeddings


if HAS_TORCH:

    class DistributionalRefinementNetwork(nn.Module):  # type: ignore[misc]
        """Refines Gamma distribution parameters from GLM base (Avanzi et al. 2023).

        Rather than predicting a single point estimate, this network outputs
        refined Gamma distribution parameters (shape α, rate β) from which
        the mean premium E[X] = α / β is derived.

        The GLM base prediction anchors the distribution.  Given a fixed
        dispersion φ, the base Gamma parameters are:

            base_shape = 1 / φ
            base_rate  = base_shape / glm_pred

        A trunk MLP learns delta adjustments to log(shape) and log(rate),
        clamped to [-2, 2] for numerical stability:

            shape_refined = base_shape * exp(delta_log_shape)
            rate_refined  = base_rate  * exp(delta_log_rate)
            pred          = shape_refined / rate_refined

        Loss function used during training:

            L = Gamma_NLL(y, shape_refined, rate_refined)
              + kl_alpha * KL(Gamma_refined || Gamma_base)

        The KL term penalises large departures from the GLM base distribution,
        providing an inductive bias toward the actuarial prior.  The
        ``kl_alpha`` attribute is stored on the module so the training loop
        can access it without additional configuration.

        The ``dist_params`` second return value has shape (batch, 2) with
        columns [shape_refined, rate_refined], enabling the training loop to
        compute the distributional loss without a second forward pass.

        Args:
            n_cont: Number of standardised continuous input features.
            category_sizes: Number of distinct categories (incl. UNKNOWN)
                per categorical column.
            embedding_dims: Embedding dimension per categorical column.
            hidden_dims: Trunk MLP hidden layer sizes.  Defaults to [128, 64].
            dropout: Dropout probability applied after each hidden activation.
            kl_alpha: Weight of the KL divergence regularisation term.
                Stored on the module for access by the training loop.
            base_dispersion: Gamma dispersion parameter φ used to derive the
                base distribution from the GLM prediction.  Clipped to a
                minimum of 1e-6 to prevent division by zero.
        """

        def __init__(
            self,
            n_cont: int,
            category_sizes: List[int],
            embedding_dims: List[int],
            hidden_dims: Optional[List[int]] = None,
            dropout: float = 0.2,
            kl_alpha: float = 0.1,
            base_dispersion: float = 1.0,
        ) -> None:
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [128, 64]

            self.kl_alpha = kl_alpha

            # Pre-compute base_shape from the dispersion parameter and register
            # as a non-trainable buffer so it moves with the model to the correct
            # device automatically.
            base_shape_val = 1.0 / max(base_dispersion, 1e-6)
            self.register_buffer("base_shape", torch.tensor(base_shape_val, dtype=torch.float32))

            self.cat_embeddings = CategoricalEmbeddings(category_sizes, embedding_dims)
            input_dim = n_cont + self.cat_embeddings.output_dim

            # Shared trunk MLP — extracts features used by both heads.
            trunk_layers: List[nn.Module] = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                trunk_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                ])
                in_dim = h_dim
            self.trunk = nn.Sequential(*trunk_layers)

            # Two separate scalar output heads — one delta per distribution parameter.
            self.head_log_shape = nn.Linear(in_dim, 1)
            self.head_log_rate = nn.Linear(in_dim, 1)

        def forward(
            self,
            x_cont: "torch.Tensor",
            x_cat: "torch.Tensor",
            glm_pred: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Forward pass returning mean premium and refined distribution parameters.

            Args:
                x_cont: Standardised continuous features, shape (batch, F_cont).
                x_cat: Categorical integer codes, shape (batch, F_cat).
                glm_pred: GLM base predictions, shape (batch,).  Clamped to a
                    minimum of 1.0 (GBP) before computing base_rate to prevent
                    division by near-zero premiums inflating the rate.

            Returns:
                Tuple of:
                    mean_pred: Gamma mean predictions (shape / rate), shape
                        (batch,).  This is the point estimate used for Tweedie
                        / Poisson-Gamma style loss functions.
                    dist_params: Stacked refined parameters, shape (batch, 2).
                        Column 0 is ``shape_refined``, column 1 is
                        ``rate_refined``.  Used by the training loop to compute
                        the full distributional loss including the KL term.
            """
            emb = self.cat_embeddings(x_cat)
            x = torch.cat([x_cont, emb], dim=1)

            # Shared feature extraction
            h = self.trunk(x)

            # Delta adjustments to the log-parameterisation, clamped for stability
            delta_log_shape = self.head_log_shape(h).squeeze(-1).clamp(-2.0, 2.0)
            delta_log_rate = self.head_log_rate(h).squeeze(-1).clamp(-2.0, 2.0)

            # Refined shape: base_shape shifted by the network's delta
            shape_refined = torch.exp(torch.log(self.base_shape) + delta_log_shape)

            # Base rate derived from GLM mean; glm_pred clamped to >= 1 GBP
            base_rate = self.base_shape / glm_pred.clamp(min=1.0)

            # Refined rate: base_rate shifted by the network's delta
            rate_refined = torch.exp(torch.log(base_rate) + delta_log_rate)

            # Gamma mean = shape / rate
            mean_pred = shape_refined / rate_refined

            # Stack parameters for the distributional loss in the training loop
            dist_params = torch.stack([shape_refined, rate_refined], dim=1)

            return mean_pred, dist_params

else:

    class DistributionalRefinementNetwork:  # type: ignore[no-redef]
        """Stub DistributionalRefinementNetwork when PyTorch is unavailable."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required for DistributionalRefinementNetwork."
            )
