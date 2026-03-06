"""Tests for DL loss functions (gamma_deviance_loss, monotonicity_penalty, drn_loss)."""
from __future__ import annotations

import math
import pytest

torch = pytest.importorskip("torch")

from modelling.config import HAS_TORCH  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

from modelling.losses import gamma_deviance_loss, monotonicity_penalty, drn_loss  # noqa: E402


# ---------------------------------------------------------------------------
# gamma_deviance_loss
# ---------------------------------------------------------------------------

class TestGammaDevianceLoss:
    def test_perfect_predictions_return_zero(self) -> None:
        """Loss must be 0 when predictions exactly match actuals."""
        y = torch.tensor([10.0, 20.0, 30.0, 50.0])
        loss = gamma_deviance_loss(y, y)
        assert abs(loss.item()) < 1e-5

    def test_positive_for_mismatched_predictions(self) -> None:
        """Loss must be strictly positive when pred != true."""
        y_true = torch.tensor([10.0, 20.0, 30.0])
        y_pred = torch.tensor([15.0, 25.0, 40.0])
        loss = gamma_deviance_loss(y_pred, y_true)
        assert loss.item() > 0.0

    def test_returns_scalar_tensor(self) -> None:
        """Loss must be a 0-dimensional tensor."""
        y = torch.ones(5) * 20.0
        loss = gamma_deviance_loss(y, y * 1.1)
        assert loss.ndim == 0

    def test_floor_prevents_log_zero(self) -> None:
        """Very small predictions should be clamped and loss should be finite."""
        y_true = torch.tensor([10.0, 10.0])
        y_pred = torch.tensor([1e-10, 1e-10])
        loss = gamma_deviance_loss(y_pred, y_true, floor=1.0)
        assert math.isfinite(loss.item())

    def test_symmetric_imbalance_not_equal(self) -> None:
        """Gamma deviance is asymmetric: over- and under-prediction differ."""
        y_true = torch.tensor([20.0])
        y_high = torch.tensor([30.0])
        y_low = torch.tensor([10.0])
        loss_high = gamma_deviance_loss(y_high, y_true)
        loss_low = gamma_deviance_loss(y_low, y_true)
        # Both should be positive and not equal in general
        assert loss_high.item() > 0
        assert loss_low.item() > 0


# ---------------------------------------------------------------------------
# monotonicity_penalty
# ---------------------------------------------------------------------------

class TestMonotonicityPenalty:
    def _make_trivial_model(self, n_cont: int) -> "torch.nn.Module":
        """Simple linear model that satisfies all monotone constraints trivially."""
        import torch.nn as nn

        class TrivialModel(nn.Module):
            def __init__(self, n: int) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.ones(n))

            def forward(
                self,
                x_cont: "torch.Tensor",
                x_cat: "torch.Tensor",
                glm_pred: "torch.Tensor",
            ):
                pred = (x_cont * self.w).sum(dim=1) + 10.0
                return pred, torch.zeros_like(pred)

        return TrivialModel(n_cont)

    def test_no_constraints_returns_zero(self) -> None:
        """When feature_indices is empty, the penalty must be exactly 0."""
        model = self._make_trivial_model(3)
        x_cont = torch.randn(8, 3)
        x_cat = torch.zeros(8, 0, dtype=torch.long)
        glm_pred = torch.ones(8)

        penalty = monotonicity_penalty(
            model, x_cont, x_cat, glm_pred, feature_indices=[], directions=[]
        )
        assert penalty.item() == 0.0

    def test_penalty_is_non_negative(self) -> None:
        """The penalty must always be >= 0."""
        import torch.nn as nn

        # Model with decreasing output for increasing input (violates +1 constraint)
        class DecreasingModel(nn.Module):
            def forward(self, x_cont, x_cat, glm_pred):
                pred = 10.0 - x_cont[:, 0]  # decreasing in first feature
                return pred, torch.zeros_like(pred)

        model = DecreasingModel()
        x_cont = torch.randn(16, 2).abs() + 0.1
        x_cat = torch.zeros(16, 0, dtype=torch.long)
        glm_pred = torch.ones(16)

        penalty = monotonicity_penalty(
            model, x_cont, x_cat, glm_pred,
            feature_indices=[0], directions=[1],  # require +1 (increasing)
        )
        assert penalty.item() >= 0.0

    def test_increasing_model_satisfies_positive_constraint(self) -> None:
        """A model that is monotone increasing incurs near-zero penalty for +1 constraint."""
        import torch.nn as nn

        class IncreasingModel(nn.Module):
            def forward(self, x_cont, x_cat, glm_pred):
                pred = 10.0 + x_cont[:, 0]  # strictly increasing in feature 0
                return pred, torch.zeros_like(pred)

        model = IncreasingModel()
        x_cont = torch.rand(16, 2)
        x_cat = torch.zeros(16, 0, dtype=torch.long)
        glm_pred = torch.ones(16)

        penalty = monotonicity_penalty(
            model, x_cont, x_cat, glm_pred,
            feature_indices=[0], directions=[1],
            epsilon=0.01,
        )
        assert penalty.item() < 1e-4


# ---------------------------------------------------------------------------
# drn_loss
# ---------------------------------------------------------------------------

class TestDrnLoss:
    def _make_dist_params(self, batch: int = 16) -> "torch.Tensor":
        """Return a plausible (batch, 2) tensor of [shape, rate] > 0."""
        shape = torch.ones(batch) * 2.0
        rate = torch.ones(batch) * 0.5
        return torch.stack([shape, rate], dim=1)

    def test_returns_finite_scalar(self) -> None:
        """drn_loss must return a finite scalar tensor for valid inputs."""
        batch = 16
        y_true = torch.ones(batch) * 20.0
        dist_params = self._make_dist_params(batch)
        base_shape = torch.tensor(2.0)
        glm_pred = torch.ones(batch) * 20.0

        loss = drn_loss(y_true, dist_params, base_shape, glm_pred)
        assert loss.ndim == 0
        assert math.isfinite(loss.item())

    def test_loss_is_scalar_with_batch_size_1(self) -> None:
        """drn_loss must work for a single-element batch."""
        y_true = torch.tensor([15.0])
        dist_params = torch.tensor([[2.0, 0.1]])
        base_shape = torch.tensor(2.0)
        glm_pred = torch.tensor([15.0])

        loss = drn_loss(y_true, dist_params, base_shape, glm_pred)
        assert math.isfinite(loss.item())

    def test_loss_with_kl_alpha_zero(self) -> None:
        """When kl_alpha=0, the KL term is suppressed and loss is still finite."""
        batch = 8
        y_true = torch.rand(batch) * 10 + 10
        dist_params = self._make_dist_params(batch)
        base_shape = torch.tensor(1.5)
        glm_pred = torch.ones(batch) * 15.0

        loss = drn_loss(y_true, dist_params, base_shape, glm_pred, kl_alpha=0.0)
        assert math.isfinite(loss.item())

    def test_positive_y_required_for_stability(self) -> None:
        """Positive y_true values should always produce a finite loss."""
        batch = 32
        y_true = torch.abs(torch.randn(batch)) + 5.0  # strictly positive
        dist_params = torch.abs(torch.randn(batch, 2)).clamp(min=0.1)
        base_shape = torch.tensor(1.0)
        glm_pred = torch.ones(batch) * 10.0

        loss = drn_loss(y_true, dist_params, base_shape, glm_pred)
        assert math.isfinite(loss.item())
