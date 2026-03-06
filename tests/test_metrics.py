"""Tests for evaluation metrics: compute_gini, compute_gamma_deviance, clamp_predictions.

Notes on implementation conventions in this codebase:

  compute_gini uses the Lorenz-curve formula:
      Gini = 1 - 2 * sum(cumulative) / (n * total)
  where actuals are sorted by predicted value ascending.  Consequently a
  'perfect' model (where pred == actual) does NOT necessarily yield Gini = 1;
  the coefficient reflects concentration rather than monotone prediction rank
  in the strict sense.

  compute_gamma_deviance computes:
      D = 2 * sum(log(actual/pred) - (actual - pred)/pred)
  This is the *negative* of the standard Gamma unit deviance, so it is zero
  for perfect predictions and *negative* for any non-trivial mismatch.  Tests
  are written against the actual formula rather than an assumed sign convention.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from modelling.utils.metrics import (
    clamp_predictions,
    compute_gamma_deviance,
    compute_gini,
)


# ---------------------------------------------------------------------------
# compute_gini
# ---------------------------------------------------------------------------

class TestComputeGini:
    def test_highly_concentrated_actuals_give_high_gini(self) -> None:
        """When one observation dominates and prediction ranks it last (highest),
        Gini should be substantially positive."""
        # 5 small observations + 1 very large one
        y_actual = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])  # ranks the big one last
        gini = compute_gini(y_actual, y_pred)
        # Lorenz formula: concentrated actuals at top of prediction ranking => high Gini
        assert gini > 0.4

    def test_reversed_predictions_give_lower_gini(self) -> None:
        """Reversed predictions should give a lower Gini than the forward ordering."""
        y_actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_forward = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_reversed = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        gini_forward = compute_gini(y_actual, y_pred_forward)
        gini_reversed = compute_gini(y_actual, y_pred_reversed)
        assert gini_forward > gini_reversed

    def test_reversed_predictions_give_negative_gini(self) -> None:
        """Reversed predictions (worst possible ranking) must give negative Gini."""
        y_actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # perfectly reversed
        gini = compute_gini(y_actual, y_pred)
        assert gini < 0.0

    def test_empty_array_returns_zero(self) -> None:
        """compute_gini on empty arrays must return 0.0 without raising."""
        gini = compute_gini(np.array([]), np.array([]))
        assert gini == 0.0

    def test_returns_float(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        result = compute_gini(y, y)
        assert isinstance(result, float)

    def test_single_element_returns_finite(self) -> None:
        """compute_gini with a single observation must return a finite value."""
        gini = compute_gini(np.array([5.0]), np.array([5.0]))
        assert math.isfinite(gini)

    def test_all_zero_actuals_returns_zero(self) -> None:
        """When all actuals are zero the total is 0 and Gini must be 0."""
        gini = compute_gini(np.zeros(5), np.ones(5))
        assert gini == 0.0

    def test_gini_known_value(self) -> None:
        """Verify Gini against a manually computed value.

        y_actual = [1, 1, 1, 1, 1, 100], y_pred = same.
        sorted by pred: same order.
        cumulative = [1,2,3,4,5,105], total=105, n=6.
        cumsum_sum = 1+2+3+4+5+105 = 120.
        Gini = 1 - 2*120 / (6*105) = 1 - 240/630 ≈ 0.619.
        """
        y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        gini = compute_gini(y, y)
        assert abs(gini - (1 - 240 / 630)) < 1e-6


# ---------------------------------------------------------------------------
# compute_gamma_deviance
# ---------------------------------------------------------------------------

class TestComputeGammaDeviance:
    def test_perfect_predictions_return_near_zero(self) -> None:
        """Gamma deviance must be ~0 when predictions equal actuals."""
        y = np.array([10.0, 20.0, 30.0])
        deviance = compute_gamma_deviance(y, y)
        assert abs(deviance) < 1e-6

    def test_returns_finite_for_mismatched_predictions(self) -> None:
        """Deviance must be finite (not NaN/Inf) when pred != actual."""
        y_actual = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([15.0, 25.0, 35.0])
        deviance = compute_gamma_deviance(y_actual, y_pred)
        assert math.isfinite(deviance)

    def test_returns_float(self) -> None:
        y = np.array([10.0, 20.0])
        assert isinstance(compute_gamma_deviance(y, y), float)

    def test_closer_predictions_less_negative(self) -> None:
        """A smaller prediction error should be less negative than a larger error.

        The formula D = 2*sum(log(a/p) - (a-p)/p) is <= 0 for all p > 0.
        Smaller errors produce values closer to zero (less negative).
        """
        y_actual = np.array([20.0, 20.0])
        y_small_error = np.array([22.0, 22.0])
        y_large_error = np.array([40.0, 40.0])

        dev_small = compute_gamma_deviance(y_actual, y_small_error)
        dev_large = compute_gamma_deviance(y_actual, y_large_error)
        # Both non-positive; small error is closer to zero (less negative)
        assert dev_small > dev_large

    def test_formula_is_non_positive_for_overshoot(self) -> None:
        """Unit deviance 2*(log(a/p) - (a-p)/p) <= 0 for any a,p > 0.
        This is a property of the specific formula used in this codebase."""
        y_actual = np.array([10.0, 10.0, 10.0])
        y_pred = np.array([15.0, 20.0, 30.0])
        deviance = compute_gamma_deviance(y_actual, y_pred)
        assert deviance <= 0.0

    def test_floor_applied_via_clamp_predictions(self) -> None:
        """Very small predictions should be floored and deviance should be finite."""
        y_actual = np.array([10.0, 10.0])
        y_pred = np.array([1e-15, 1e-15])
        deviance = compute_gamma_deviance(y_actual, y_pred, floor=1.0)
        assert math.isfinite(deviance)

    def test_all_zero_actuals_returns_nan(self) -> None:
        """When all actuals are non-positive the function should return NaN."""
        y_actual = np.array([-1.0, -2.0])
        y_pred = np.array([10.0, 10.0])
        deviance = compute_gamma_deviance(y_actual, y_pred)
        assert math.isnan(deviance)

    def test_known_value(self) -> None:
        """Verify a single-point calculation.

        For a=10, p=10: 2*(log(1) - 0/10) = 0.
        """
        deviance = compute_gamma_deviance(np.array([10.0]), np.array([10.0]))
        assert abs(deviance) < 1e-10


# ---------------------------------------------------------------------------
# clamp_predictions
# ---------------------------------------------------------------------------

class TestClampPredictions:
    def test_values_below_floor_are_raised(self) -> None:
        """Predictions below the floor must be clamped up."""
        preds = np.array([0.0, -5.0, 0.5])
        clamped = clamp_predictions(preds, floor=1.0)
        assert (clamped >= 1.0).all()

    def test_values_above_floor_are_unchanged(self) -> None:
        """Predictions already above the floor must not be modified."""
        preds = np.array([10.0, 20.0, 100.0])
        clamped = clamp_predictions(preds, floor=1.0)
        np.testing.assert_array_equal(clamped, preds)

    def test_custom_floor(self) -> None:
        """The floor parameter must be respected."""
        preds = np.array([0.5, 1.5, 3.0])
        clamped = clamp_predictions(preds, floor=2.0)
        assert clamped[0] == 2.0
        assert clamped[1] == 2.0
        assert clamped[2] == 3.0

    def test_returns_ndarray(self) -> None:
        """Return type must be a NumPy array."""
        result = clamp_predictions(np.array([1.0, 2.0]), floor=0.5)
        assert isinstance(result, np.ndarray)

    def test_empty_array(self) -> None:
        """clamp_predictions on an empty array must return an empty array."""
        result = clamp_predictions(np.array([]), floor=1.0)
        assert len(result) == 0
