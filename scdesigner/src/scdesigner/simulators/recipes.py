"""Simulators composed from a marginal, a copula, and spatial terms."""

import copy
from typing import Mapping, Optional, Tuple

from ..base.copula import Copula
from ..copulas.standard_copula import StandardCopula
from ..data.spatial import SpatialDesign, SpatialTerm
from ..distributions.negbin import NegBin
from .output import ObservedMaxCap
from .scd3 import SCD3Simulator


def spatial_copula(
    marginal: NegBin,
    terms: Mapping[str, SpatialTerm],
    *,
    copula: Optional[Copula] = None,
    spatial_cols: Tuple[str, str] = ("spatial1", "spatial2"),
    cap_at_observed_max: bool = False,
) -> SCD3Simulator:
    """:class:`SCD3Simulator` with spatial basis columns added

    This is a light wrapper of the SCD3Simulator that adds spatial design
    columns. Philosophically, we think of this as a part of a "recipe" approach
    to combining simulation properties. Earlier, we layered penalties onto the
    marginal model. At this step, we layer on spatial design matrices.

    Parameters
    ----------
    marginal : NegBin
        Copied; its formula and penalties are the nonspatial base.
    terms : mapping of str to SpatialTerm
        Term per marginal parameter; see
        :class:`~scdesigner.data.spatial.SpatialDesign`.
    copula : Copula, optional
        Copied. Defaults to ``StandardCopula("~ 1")``.
    spatial_cols : tuple of str
    cap_at_observed_max : bool
        Cap sampled counts with
        :class:`~scdesigner.simulators.output.ObservedMaxCap`. In earlier
        experiments, we found that this prevented extreme outliers that happen
        in high-dimensional negbin models.

    Examples
    --------
    >>> from scdesigner.data import GPBasis, SpatialTerm
    >>> from scdesigner.distributions import NegBin
    >>> from scdesigner.distributions.penalties import QuadraticPenalty
    >>> term = SpatialTerm(GPBasis(df=15, standardize=True), QuadraticPenalty(lam=1.0))
    >>> sim = spatial_copula(NegBin({"mean": "~ 1", "dispersion": "~ 1"}), {"mean": term})
    >>> sim.fit(adata)  # doctest: +SKIP
    """
    if type(marginal) is not NegBin:
        raise NotImplementedError("spatial_copula currently supports only NegBin marginals.")

    marginal = copy.deepcopy(marginal)
    copula = StandardCopula("~ 1") if copula is None else copy.deepcopy(copula)
    return SCD3Simulator(
        marginal,
        copula,
        observation_transform=SpatialDesign(marginal.formula, terms, spatial_cols),
        output_transform=ObservedMaxCap() if cap_at_observed_max else None,
    )
