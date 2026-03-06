"""CANN-GBM — Combined Actuarial Neural Network with GBM base predictions."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..config import HAS_TORCH, torch, nn
from .shared import CategoricalEmbeddings


if HAS_TORCH:

    class CANNGBM(nn.Module):  # type: ignore[misc]
        """CANN with GBM (CatBoost) base instead of GLM base.

        Architecture identical to CANN: MLP residual on base predictions.

            pred = base_pred * exp(clamp(nn_residual, -2, 2))

        The base_pred comes from a pre-trained CatBoost model rather than a
        GLM.  The distinction is handled entirely at the training level — the
        caller passes GBM predictions via the ``glm_pred`` argument, which
        acts as the generic ``base_pred`` input.  Having a separate class
        allows independent hyperparameter configuration (e.g. shallower
        hidden_dims, lower dropout) tuned specifically for a GBM residual.

        Clamping the residual to [-2, 2] limits the neural correction to a
        factor of exp(2) ≈ 7.4x in either direction, preventing the network
        from overriding the GBM signal entirely.

        Args:
            n_cont: Number of standardised continuous input features.
            category_sizes: Number of distinct categories (incl. UNKNOWN)
                per categorical column.
            embedding_dims: Embedding dimension per categorical column.
            hidden_dims: List of hidden layer sizes for the MLP residual
                network.  Defaults to [128, 64].
            dropout: Dropout probability applied after each hidden activation.
        """

        def __init__(
            self,
            n_cont: int,
            category_sizes: List[int],
            embedding_dims: List[int],
            hidden_dims: Optional[List[int]] = None,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [128, 64]

            self.cat_embeddings = CategoricalEmbeddings(category_sizes, embedding_dims)
            input_dim = n_cont + self.cat_embeddings.output_dim

            layers: List[nn.Module] = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                ])
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(
            self,
            x_cont: "torch.Tensor",
            x_cat: "torch.Tensor",
            glm_pred: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Forward pass returning final premium prediction and NN residual.

            Args:
                x_cont: Standardised continuous features, shape (batch, F_cont).
                x_cat: Categorical integer codes, shape (batch, F_cat).
                glm_pred: GBM base predictions, shape (batch,).  Named
                    ``glm_pred`` for API compatibility with the shared training
                    loop; the caller supplies CatBoost predictions here.

            Returns:
                Tuple of:
                    pred: Final premium predictions, shape (batch,).
                        Computed as ``base_pred * exp(clamp(residual, -2, 2))``.
                    nn_residual: Raw NN output (log-scale correction),
                        shape (batch,).  Useful for diagnostics and
                        regularisation of the residual magnitude.
            """
            emb = self.cat_embeddings(x_cat)
            x = torch.cat([x_cont, emb], dim=1)
            nn_residual = self.mlp(x).squeeze(-1)
            clamped = nn_residual.clamp(min=-2.0, max=2.0)
            pred = glm_pred * torch.exp(clamped)
            return pred, nn_residual

else:

    class CANNGBM:  # type: ignore[no-redef]
        """Stub CANNGBM when PyTorch is unavailable."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for CANNGBM.")
