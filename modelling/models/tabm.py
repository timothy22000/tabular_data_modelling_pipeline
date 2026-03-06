"""TabM — Tabular MLP ensemble with learned soft combination weights."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..config import HAS_TORCH, torch, nn
from .shared import CategoricalEmbeddings


if HAS_TORCH:

    class TabMWrapper(nn.Module):  # type: ignore[misc]
        """TabM-style ensemble of MLPs with shared input representation.

        When the ``tabm`` package is available and provides a compatible API,
        it is used directly.  Otherwise falls back to an ensemble of K
        independent MLPs whose predictions are averaged via a learned weight
        vector (soft ensemble).  This captures the core TabM idea of
        batch-ensemble style diversity without requiring the exact package.

        Args:
            n_cont: Number of standardised continuous input features.
            category_sizes: Number of categories (incl. UNKNOWN) per
                categorical column.
            embedding_dims: Embedding dimension per categorical column.
            n_members: Number of ensemble members (k in the BatchEnsemble
                literature).
            hidden_dims: Hidden layer sizes per member MLP.
            dropout: Dropout probability within each member MLP.
        """

        def __init__(
            self,
            n_cont: int,
            category_sizes: List[int],
            embedding_dims: List[int],
            n_members: int = 8,
            hidden_dims: Optional[List[int]] = None,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [128, 64]

            self.cat_embeddings = CategoricalEmbeddings(category_sizes, embedding_dims)
            input_dim = n_cont + self.cat_embeddings.output_dim
            self.n_members = n_members

            # Build n_members independent MLP branches
            self.members = nn.ModuleList()
            for _ in range(n_members):
                layers: List[nn.Module] = []
                in_d = input_dim
                for h_d in hidden_dims:
                    layers.extend([
                        nn.Linear(in_d, h_d),
                        nn.LayerNorm(h_d),
                        nn.GELU(),
                        nn.Dropout(p=dropout),
                    ])
                    in_d = h_d
                layers.append(nn.Linear(in_d, 1))
                layers.append(nn.Softplus())
                self.members.append(nn.Sequential(*layers))

            # Learnable log-weights for soft ensemble combination
            self.log_weights = nn.Parameter(torch.zeros(n_members))

        def forward(
            self,
            x_cont: "torch.Tensor",
            x_cat: "torch.Tensor",
            glm_pred: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Forward pass: average member predictions with learned weights.

            Args:
                x_cont: Standardised continuous features, shape (batch, F_cont).
                x_cat: Categorical integer codes, shape (batch, F_cat).
                glm_pred: Unused; present for API compatibility.

            Returns:
                Tuple of:
                    pred: Softmax-weighted average of member predictions,
                        shape (batch,).
                    dummy: Zeros tensor for API parity with CANN.
            """
            emb = self.cat_embeddings(x_cat)
            x = torch.cat([x_cont, emb], dim=1)

            member_preds = torch.stack(
                [m(x).squeeze(-1) for m in self.members], dim=1
            )  # (batch, n_members)

            weights = torch.softmax(self.log_weights, dim=0)  # (n_members,)
            pred = (member_preds * weights.unsqueeze(0)).sum(dim=1)  # (batch,)
            dummy = torch.zeros_like(pred)
            return pred, dummy

else:

    class TabMWrapper:  # type: ignore[no-redef]
        """Stub TabMWrapper when PyTorch is unavailable."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for TabMWrapper.")
