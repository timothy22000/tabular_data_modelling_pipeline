"""FT-Transformer — Feature-Tokenizer Transformer for tabular data."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..config import HAS_TORCH, torch, nn


if HAS_TORCH:

    class FeatureTokenizer(nn.Module):  # type: ignore[misc]
        """Per-feature tokeniser for the FT-Transformer.

        Each continuous feature is projected to a d_model-dimensional token
        via a learned linear layer.  Each categorical feature uses an
        nn.Embedding of size d_model.  A learnable [CLS] token is prepended
        to the sequence for downstream classification/regression.

        Args:
            n_cont: Number of standardised continuous features.
            category_sizes: Number of categories (incl. UNKNOWN) per
                categorical feature.
            d_model: Token embedding dimension used throughout the transformer.
        """

        def __init__(
            self,
            n_cont: int,
            category_sizes: List[int],
            d_model: int = 64,
        ) -> None:
            super().__init__()
            self.d_model = d_model

            # One Linear per continuous feature: R^1 -> R^d_model
            self.cont_projections = nn.ModuleList(
                [nn.Linear(1, d_model) for _ in range(n_cont)]
            )

            # One Embedding per categorical feature: int -> R^d_model
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(n_cats, d_model) for n_cats in category_sizes]
            )

            # Learnable [CLS] token prepended to the token sequence
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        @property
        def n_tokens(self) -> int:
            """Total number of tokens including [CLS]."""
            return 1 + len(self.cont_projections) + len(self.cat_embeddings)

        def forward(
            self, x_cont: "torch.Tensor", x_cat: "torch.Tensor"
        ) -> "torch.Tensor":
            """Tokenise all features and prepend [CLS].

            Args:
                x_cont: Continuous features, shape (batch, F_cont).
                x_cat: Categorical codes, shape (batch, F_cat).

            Returns:
                Token sequence, shape (batch, n_tokens, d_model).
            """
            tokens: List["torch.Tensor"] = []

            # Continuous tokens: (batch, 1) -> (batch, 1, d_model)
            for i, proj in enumerate(self.cont_projections):
                feat = x_cont[:, i:i + 1]  # (batch, 1)
                tokens.append(proj(feat).unsqueeze(1))  # (batch, 1, d_model)

            # Categorical tokens: int -> (batch, 1, d_model)
            for j, emb in enumerate(self.cat_embeddings):
                tokens.append(emb(x_cat[:, j]).unsqueeze(1))

            # Stack all feature tokens
            token_seq = torch.cat(tokens, dim=1)  # (batch, F_cont+F_cat, d_model)

            # Prepend [CLS] token
            batch_size = x_cont.size(0)
            cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
            token_seq = torch.cat([cls, token_seq], dim=1)  # (batch, n_tokens, d_model)

            return token_seq

    class FTTransformer(nn.Module):  # type: ignore[misc]
        """Feature-Tokenizer Transformer for tabular premium prediction.

        Architecture following Gorishniy et al. (2021) "Revisiting Deep
        Learning Models for Tabular Data":
          1. FeatureTokenizer: convert each feature into a d_model-dim token.
          2. TransformerEncoder: self-attention over the token sequence.
          3. Head MLP: map the [CLS] token representation to a premium
             prediction via Softplus to ensure positive output.

        Args:
            n_cont: Number of standardised continuous input features.
            category_sizes: Number of categories (incl. UNKNOWN) per
                categorical column.
            d_model: Transformer token/embedding dimension.
            n_heads: Number of attention heads (must divide d_model).
            n_layers: Number of TransformerEncoderLayer blocks.
            dropout: Dropout probability within attention and feed-forward.
            ffn_factor: Feed-forward hidden dimension as a multiple of d_model.
        """

        def __init__(
            self,
            n_cont: int,
            category_sizes: List[int],
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 3,
            dropout: float = 0.1,
            ffn_factor: int = 4,
        ) -> None:
            super().__init__()
            self.tokenizer = FeatureTokenizer(n_cont, category_sizes, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * ffn_factor,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN for training stability
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
                norm=nn.LayerNorm(d_model),
            )

            # Regression head: [CLS] -> hidden -> output -> Softplus
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(d_model // 2, 1),
                nn.Softplus(),  # Guarantees positive premium output
            )

        def forward(
            self,
            x_cont: "torch.Tensor",
            x_cat: "torch.Tensor",
            glm_pred: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Forward pass.

            Args:
                x_cont: Standardised continuous features, shape (batch, F_cont).
                x_cat: Categorical integer codes, shape (batch, F_cat).
                glm_pred: Unused; present for API compatibility with CANN
                    training loop.

            Returns:
                Tuple of:
                    pred: Premium predictions, shape (batch,). Softplus-
                        activated, guaranteed positive.
                    dummy: Zeros tensor of the same shape (for API parity
                        with CANN which returns (pred, residual)).
            """
            tokens = self.tokenizer(x_cont, x_cat)          # (batch, n_tokens, d_model)
            encoded = self.transformer(tokens)               # (batch, n_tokens, d_model)
            cls_out = encoded[:, 0, :]                       # (batch, d_model)
            pred = self.head(cls_out).squeeze(-1)            # (batch,)
            dummy = torch.zeros_like(pred)
            return pred, dummy

else:

    class FeatureTokenizer:  # type: ignore[no-redef]
        """Stub FeatureTokenizer when PyTorch is unavailable."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for FeatureTokenizer.")

    class FTTransformer:  # type: ignore[no-redef]
        """Stub FTTransformer when PyTorch is unavailable."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for FTTransformer.")
