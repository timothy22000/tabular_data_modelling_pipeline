"""LocalGLMnet — instance-specific GLM coefficients via neural network."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..config import HAS_TORCH, torch, nn
from .shared import CategoricalEmbeddings


if HAS_TORCH:

    class LocalGLMnet(nn.Module):  # type: ignore[misc]
        """Instance-specific GLM coefficients via neural network (Richman & Wüthrich 2023).

        A neural network generates per-sample coefficients β_k(x) for each
        continuous feature x_k.  The final prediction combines the GLM base
        with a learned feature-specific adjustment:

            pred = glm_pred * exp(clamp(Σ_k β_k(x) · x_k, -3, 3))

        The inner product Σ β_k · x_k is the instance-specific log-correction.
        Clamping to [-3, 3] keeps predictions in the range
        [glm_pred / exp(3), glm_pred * exp(3)] — a factor of ~20x in either
        direction — which is wide enough to be expressive while avoiding the
        premium blow-up that would occur with an unclamped exponent.

        The ``coeff_reg`` attribute exposes the L2 regularisation weight for
        the coefficient matrix.  The training loop should add::

            coeff_reg * coeffs.pow(2).mean()

        to the loss when this model is used, to discourage large per-feature
        corrections.

        Args:
            n_cont: Number of standardised continuous input features.
            category_sizes: Number of distinct categories (incl. UNKNOWN)
                per categorical column.
            embedding_dims: Embedding dimension per categorical column.
            hidden_dims: MLP hidden layer sizes for the coefficient network.
                Defaults to [128, 64].
            dropout: Dropout probability applied after each hidden activation.
            coeff_reg: L2 regularisation weight for the coefficient tensor.
                Stored on the module so training loops can access it via
                ``model.coeff_reg``.  Does not affect the forward pass itself.
        """

        def __init__(
            self,
            n_cont: int,
            category_sizes: List[int],
            embedding_dims: List[int],
            hidden_dims: Optional[List[int]] = None,
            dropout: float = 0.2,
            coeff_reg: float = 0.01,
        ) -> None:
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [128, 64]

            self.n_cont = n_cont
            self.coeff_reg = coeff_reg

            self.cat_embeddings = CategoricalEmbeddings(category_sizes, embedding_dims)
            input_dim = n_cont + self.cat_embeddings.output_dim

            # MLP that maps the full feature representation to n_cont coefficients.
            # Output dimension equals n_cont so that β_k(x) · x_k is well defined
            # for each of the k continuous features.
            coeff_layers: List[nn.Module] = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                coeff_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                ])
                in_dim = h_dim
            final_linear = nn.Linear(in_dim, n_cont)
            # Initialize final layer near zero so the model starts close to
            # the GLM base prediction (adjustment ≈ 0 → exp(0) = 1).
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
            coeff_layers.append(final_linear)
            self.coeff_net = nn.Sequential(*coeff_layers)

        def forward(
            self,
            x_cont: "torch.Tensor",
            x_cat: "torch.Tensor",
            glm_pred: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Forward pass returning final premium prediction and coefficient matrix.

            Args:
                x_cont: Standardised continuous features, shape (batch, F_cont).
                x_cat: Categorical integer codes, shape (batch, F_cat).
                glm_pred: GLM base predictions, shape (batch,).

            Returns:
                Tuple of:
                    pred: Final premium predictions, shape (batch,).
                        Computed as
                        ``glm_pred * exp(clamp(Σ β_k(x) · x_k, -3, 3))``.
                    coeffs: Per-sample feature coefficients β(x), shape
                        (batch, n_cont).  Return value is useful for L2
                        regularisation in the training loop and for
                        post-hoc interpretability analysis.
            """
            emb = self.cat_embeddings(x_cat)
            x = torch.cat([x_cont, emb], dim=1)

            # β(x): (batch, n_cont) — one coefficient per continuous feature
            coeffs = self.coeff_net(x)

            # Instance-specific log-adjustment: Σ_k β_k(x) · x_k
            adjustment = (coeffs * x_cont).sum(dim=1)  # (batch,)

            pred = glm_pred * torch.exp(adjustment.clamp(min=-1.0, max=1.0))
            return pred, coeffs

else:

    class LocalGLMnet:  # type: ignore[no-redef]
        """Stub LocalGLMnet when PyTorch is unavailable."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for LocalGLMnet.")
