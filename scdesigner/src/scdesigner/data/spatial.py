"""Spatial basis columns for marginal-parameter formulas.

:class:`SpatialDesign` fits spatial bases for parameters with
:class:`SpatialTerm`s, appends the ``K`` bases columns to the `.obs` as
``sp_<parameter>_0, ..., sp_<parameter>_<K-1>``, and adds them to the
parameter's formula. A penalized term adds a
:class:`~scdesigner.distributions.penalties.CoefficientPenalty` to those columns
with ``d = basis.penalty_diag``.
"""

import copy
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..distributions.penalties import CoefficientPenalty, QuadraticPenalty
from .basis import SpatialBasis


@dataclass(frozen=True)
class SpatialTerm:
    """Unfitted spatial basis and optional penalty on its coefficients.

    Parameters
    ----------
    basis : SpatialBasis
        Template
    penalty : QuadraticPenalty, optional
        ``None`` leaves the basis coefficients unpenalized.
    """

    basis: SpatialBasis
    penalty: Optional[QuadraticPenalty] = None


class SpatialDesign:
    """Extend .obs to include spatial basis columns.

    Parameters
    ----------
    base_formula : mapping of str to str
        Formula per marginal parameter, before spatial columns.
    terms : mapping of str to SpatialTerm
        Term per parameter. Parameters without one are unchanged.
    spatial_cols : tuple of str
        Coordinate columns of the observations.

    Attributes
    ----------
    formula : dict of str to str
        ``base_formula`` with the generated columns of each fitted term appended.
    penalties : tuple of CoefficientPenalty
        One per penalized term.
    bases : dict of str to SpatialBasis
        Fitted basis per term.
    """

    def __init__(
        self,
        base_formula: Mapping[str, str],
        terms: Mapping[str, SpatialTerm],
        spatial_cols: Tuple[str, str] = ("spatial1", "spatial2"),
    ):
        self.base_formula = dict(base_formula)
        self.terms = copy.deepcopy(dict(terms))
        self.spatial_cols = tuple(spatial_cols)
        self.formula = dict(self.base_formula)
        self.penalties = ()
        self.bases = {}
        self._columns = None

    @property
    def is_fitted(self) -> bool:
        return self._columns is not None

    def fit_transform(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Fit the spatial basis on coordinates in ``obs``.

        Returns :meth:`transform` of ``obs``.
        """
        bases = {name: copy.deepcopy(term.basis) for name, term in self.terms.items()}
        if bases:
            coords = self._coords(obs)
            for basis in bases.values():
                basis.fit(coords)

        # get the new column names
        columns = {
            name: tuple(f"sp_{name}_{k}" for k in range(len(basis.penalty_diag)))
            for name, basis in bases.items()
        }

        # create the formula
        formula = dict(self.base_formula)
        for name, generated in columns.items():
            formula[name] = " + ".join([formula[name], *generated])

        # build the penalty
        self.formula = formula
        self.penalties = tuple(
            CoefficientPenalty(name, columns[name], bases[name].penalty_diag, term.penalty)
            for name, term in self.terms.items()
            if term.penalty is not None
        )
        self.bases, self._columns = bases, columns
        return self.transform(obs)

    def transform(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Copy of ``obs`` with the fitted basis columns appended."""
        if not self.is_fitted:
            raise RuntimeError("SpatialDesign.transform requires a successful fit.")
        if not self.bases:
            return obs.copy()
        coords = self._coords(obs)
        generated = {}
        for name, basis in self.bases.items():
            generated.update(zip(self._columns[name], basis.transform(coords).T))
        return obs.assign(**generated)

    def _coords(self, obs: pd.DataFrame) -> np.ndarray:
        return obs[list(self.spatial_cols)].to_numpy(dtype=np.float64)
