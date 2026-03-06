"""Shared neural network components used by multiple model architectures."""
from __future__ import annotations

from typing import Any, List

from ..config import HAS_TORCH, torch, nn


if HAS_TORCH:

    class CategoricalEmbeddings(nn.Module):  # type: ignore[misc]
        """Embedding lookup table for all categorical features.

        Creates one ``nn.Embedding`` per categorical column, looks up each,
        and concatenates the resulting embedding vectors into a single flat
        representation.  The zero-index (UNKNOWN token) is treated as any
        other category — no special masking is applied.

        Args:
            category_sizes: Number of distinct categories (including UNKNOWN)
                per categorical column.  Each value equals
                ``max(encoding_code) + 1``.
            embedding_dims: Embedding dimension per categorical column.
                Must have the same length as ``category_sizes``.
        """

        def __init__(
            self,
            category_sizes: List[int],
            embedding_dims: List[int],
        ) -> None:
            super().__init__()
            assert len(category_sizes) == len(embedding_dims), (
                "category_sizes and embedding_dims must have equal length"
            )
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(n_cats, dim, padding_idx=None)
                    for n_cats, dim in zip(category_sizes, embedding_dims)
                ]
            )
            self.output_dim: int = sum(embedding_dims)

        def forward(self, x_cat: "torch.Tensor") -> "torch.Tensor":
            """Look up and concatenate embeddings for all categorical columns.

            Args:
                x_cat: Long tensor of shape (batch, n_categoricals).

            Returns:
                Float tensor of shape (batch, sum_of_embedding_dims).
            """
            embedded: List["torch.Tensor"] = [
                emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)
            ]
            if not embedded:
                return torch.zeros(x_cat.size(0), 0, dtype=torch.float32, device=x_cat.device)
            return torch.cat(embedded, dim=1)

else:
    class CategoricalEmbeddings:  # type: ignore[no-redef]
        """Stub CategoricalEmbeddings when PyTorch is unavailable."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for CategoricalEmbeddings.")
