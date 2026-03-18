"""Probability calibration for win probability model (ADR-004).

Applies Platt scaling (logistic regression on raw logits) to ensure
P(win)=0.70 means the player actually wins 70% of games at that
predicted probability level.

Without calibration, raw sigmoid outputs from neural networks are
typically overconfident — Platt scaling corrects this with a learned
affine transform on the logit space.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PlattCalibrator:
    """Platt scaling calibration for binary probability outputs.

    Fits: P_calibrated = 1 / (1 + exp(a*f + b))
    where f is the raw model logit.

    This is equivalent to temperature scaling with a bias term,
    correcting both miscalibrated confidence and shifted decision
    boundaries.
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0
        self._fitted: bool = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 1000,
    ) -> "PlattCalibrator":
        """Fit Platt scaling parameters from validation data.

        Uses sklearn's LogisticRegression (L-BFGS solver) which is
        numerically stable and handles near-separable data gracefully.

        The fitted model is: P_cal = 1 / (1 + exp(a*f + b))
        which maps to LogisticRegression's: P = 1 / (1 + exp(-(w*f + b)))
        so a = -w, b_platt = -b_lr.

        Args:
            logits: Raw model logits (before sigmoid), shape (N,).
            labels: Binary ground truth labels, shape (N,).
            max_iter: Maximum optimization iterations.

        Returns:
            Self for chaining.
        """
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(
            C=1e10,  # Near-zero regularization (pure Platt scaling)
            solver="lbfgs",
            max_iter=max_iter,
        )
        lr.fit(logits.reshape(-1, 1), labels)

        # Convert sklearn convention to Platt convention
        # sklearn: P = sigmoid(w*x + b)
        # Platt:   P = sigmoid(-(a*x + b)) = 1/(1+exp(a*x+b))
        self.a = -float(lr.coef_[0, 0])
        self.b = -float(lr.intercept_[0])
        self._fitted = True

        calibrated = self.calibrate_logits(logits)
        ece = self._expected_calibration_error(calibrated, labels)
        logger.info(
            "Platt scaling fit: a=%.4f, b=%.4f, ECE=%.4f", self.a, self.b, ece,
        )

        return self

    @property
    def fitted(self) -> bool:
        """Whether calibration parameters have been fit."""
        return self._fitted

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration to raw logits.

        Args:
            logits: Raw model logits, any shape.

        Returns:
            Calibrated probabilities in [0, 1].
        """
        exp_arg = np.clip(self.a * logits + self.b, -500, 500)
        return 1.0 / (1.0 + np.exp(exp_arg))

    def calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to sigmoid probabilities.

        Converts probs back to logits, applies Platt scaling.

        Args:
            probs: Sigmoid probabilities in (0, 1).

        Returns:
            Calibrated probabilities.
        """
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs / (1 - probs))
        return self.calibrate_logits(logits)

    @staticmethod
    def _expected_calibration_error(
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Bins predictions into deciles and computes weighted average
        of |predicted - actual| per bin.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)
        return float(ece)

    def reliability_diagram(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> list[dict]:
        """Compute reliability diagram data for verification.

        Args:
            probs: Calibrated probabilities.
            labels: Binary ground truth.
            n_bins: Number of bins.

        Returns:
            List of dicts with bin_center, predicted, actual, count.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            count = int(mask.sum())
            if count == 0:
                continue
            bins.append({
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "predicted": float(probs[mask].mean()),
                "actual": float(labels[mask].mean()),
                "count": count,
            })
        return bins

    def to_dict(self) -> dict:
        """Serialize calibration parameters."""
        return {"a": self.a, "b": self.b, "fitted": self._fitted}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        """Deserialize calibration parameters."""
        cal = cls()
        cal.a = d["a"]
        cal.b = d["b"]
        cal._fitted = d.get("fitted", True)
        return cal

    def save(self, path: Path) -> None:
        """Save calibration parameters to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        logger.info("Calibrator saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "PlattCalibrator":
        """Load calibration parameters from JSON file."""
        with open(path) as f:
            cal = cls.from_dict(json.load(f))
        logger.info("Calibrator loaded: a=%.4f, b=%.4f", cal.a, cal.b)
        return cal
