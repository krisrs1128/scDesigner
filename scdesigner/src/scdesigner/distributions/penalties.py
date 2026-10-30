"""Quadratic penalties on named rows of marginal coefficient matrices.

These helper functions help regularize high-dimensional parameter functions
(e.g., when derived from a large spatial basis). For coefficients ``B`` (design
columns x features), rows ``r_k`` selected by name, and diagonal ``d``,

    P(B) = lam * sum_j w_j * sum_k d_k * B[r_k, j]**2,

where ``w_j = 1 / max(m_j, 0.01)`` for feature means ``m`` under
``"inverse_mean"`` weighting, and ``w_j = 1`` under ``"none"``.
"""

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence, Union

import numpy as np
import torch

MEAN_FLOOR = 0.01


@dataclass(frozen=True)
class QuadraticPenalty:
    """Strength ``lam >= 0`` and feature weights ``w`` of ``P``.

    Parameters
    ----------
    lam : float
    weighting : {"inverse_mean", "none"}
    """

    lam: float = 1.0
    weighting: Literal["inverse_mean", "none"] = "inverse_mean"


class CoefficientPenalty:
    """``P`` on named design columns of one marginal parameter.

    Parameters
    ----------
    parameter : str
        Penalized parameter, e.g. ``"mean"``.
    columns : sequence of str
        Penalized design columns. Matched by name, so their position in the
        design is irrelevant.
    diagonal : array-like of shape (len(columns),)
        Nonnegative ``d_k`` for each column.
    penalty : QuadraticPenalty

    Examples
    --------
    >>> spec = CoefficientPenalty("mean", ["x"], [1.0], QuadraticPenalty(lam=0.1))
    >>> spec.columns
    ('x',)
    """

    def __init__(
        self,
        parameter: str,
        columns: Sequence[str],
        diagonal: Union[Sequence[float], np.ndarray],
        penalty: QuadraticPenalty,
    ):
        self.parameter = parameter
        self.columns = tuple(columns)
        self.diagonal = np.array(diagonal, dtype=np.float64)
        self.diagonal.setflags(write=False)
        self.penalty = penalty

    @property
    def uses_feature_means(self) -> bool:
        """Whether :meth:`bind` needs ``feature_means``."""
        return self.penalty.weighting == "inverse_mean"

    def bind(
        self,
        predictor_names: Mapping[str, Sequence[str]],
        coefs: Mapping[str, torch.Tensor],
        feature_means: Optional[np.ndarray] = None,
    ) -> "BoundCoefficientPenalty":
        """Bind to coefficients

        Parameters
        ----------
        predictor_names : mapping of str to sequence of str
            Design-column names per parameter.
        coefs : mapping of str to torch.Tensor
            Coefficients per parameter, shape (n_design_columns, n_features).
        feature_means : ndarray of shape (n_features,), optional
            ``m``; used only under inverse-mean weighting.
        """
        coefficient = coefs[self.parameter]
        names = list(predictor_names[self.parameter])
        rows = [names.index(column) for column in self.columns]

        def as_tensor(values, dtype=coefficient.dtype):
            return torch.tensor(values, dtype=dtype, device=coefficient.device)

        return BoundCoefficientPenalty(
            coefficient=coefficient,
            rows=as_tensor(rows, dtype=torch.long),
            diagonal=as_tensor(self.diagonal),
            weights=as_tensor(self._feature_weights(feature_means, coefficient.shape[1])),
            lam=self.penalty.lam,
        )

    def _feature_weights(self, feature_means, n_features: int) -> np.ndarray:
        if not self.uses_feature_means:
            return np.ones(n_features)
        means = np.asarray(feature_means, dtype=np.float64)
        return 1.0 / np.maximum(means, MEAN_FLOOR)


@dataclass(frozen=True, eq=False)
class BoundCoefficientPenalty:
    """`Bind a penalty `P`` to a coefficient tensor by reference."""

    coefficient: torch.Tensor
    rows: torch.Tensor
    diagonal: torch.Tensor
    weights: torch.Tensor
    lam: float

    def value(self) -> torch.Tensor:
        """``P`` at the current coefficients"""
        selected = self.coefficient[self.rows]
        return self.lam * (self.diagonal @ selected.square() @ self.weights)
