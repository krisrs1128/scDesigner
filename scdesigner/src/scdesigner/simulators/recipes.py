"""Simulators composed from a marginal, a copula, and spatial terms."""

import copy
from typing import Mapping, Optional, Sequence, Tuple

from ..base.copula import Copula
from ..copulas.standard_copula import StandardCopula
from ..data.basis import GPBasis, TPSBasis
from ..data.spatial import SpatialDesign, SpatialTerm
from ..distributions.negbin import NegBin
from ..distributions.penalized_negbin import PenalizedNegBin
from ..distributions.penalties import CoefficientPenalty, QuadraticPenalty
from .output import ObservedMaxCap
from .scd3 import SCD3Simulator

BASES = {"gp": GPBasis, "tps": TPSBasis}


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


def SpatialNegBinCopula(
    mean_formula: str = "~ 1",
    dispersion_formula: str = "~ 1",
    copula_formula: str = "~ 1",
    *,
    mean_df: int = 15,
    disp_df: Optional[int] = None,
    basis: str = "gp",
    standardize: bool = True,
    lam: float = 1.0,
    lam_disp: float = 1.0,
    spatial_cols: Tuple[str, str] = ("spatial1", "spatial2"),
    cap_at_observed_max: bool = True,
    estimator=None,
    top_k: Optional[int] = None,
    device=None,
    **basis_kwargs,
) -> SCD3Simulator:
    """A negative binomial simulator suited to spatial data

    This is a convenience wrapper of `spatial_copula` specialized to negative
    binomial marginal models. It automatically constructs the design matrix and
    applies regularization terms in a way that suits spatial count data.

    Internally, we construct ``SpatialTerm(basis(df, standardize,
    **basis_kwargs), QuadraticPenalty(lam, "inverse_mean"))`` objects, which are
    then fed in to :func:`spatial_copula`.

    Parameters
    ----------
    mean_formula, dispersion_formula : str
        Nonspatial part of each parameter's formula.
    copula_formula : str
    mean_df, disp_df : int
        Level of complexity for the mean and dispersion functions. Larger ``df``
        corresponds to higher complexity.  Setting ``disp_df=None`` means that
        the dispersion does not depend on the spatial coordinate.
    basis : {"gp", "tps"}
       Whether to use a Gaussian process  (:class:`GPBasis`) or thin-plate
       spline (:class:`TPSBasis`) basis for the mean and (optionally) dispersion
       parameters
    standardize : bool
    lam, lam_disp : float
        Penalty strength for the mean and dispersion terms.
    spatial_cols : tuple of str
    cap_at_observed_max : bool
    estimator, top_k
        Passed to :class:`StandardCopula`.
    device : torch.device or str, optional
        Marginal device.
    **basis_kwargs
        Passed to both bases, e.g. ``n_landmarks`` or ``length_scale``.

    Examples
    --------
    >>> sim = SpatialNegBinCopula("~ 0 + cell_type", basis="tps", disp_df=5)
    >>> sim.fit(adata)  # doctest: +SKIP
    """

    def term(df, strength):
        return SpatialTerm(
            BASES[basis](df=df, standardize=standardize, **basis_kwargs),
            QuadraticPenalty(lam=strength, weighting="inverse_mean"),
        )

    terms = {"mean": term(mean_df, lam)}
    if disp_df is not None:
        terms["dispersion"] = term(disp_df, lam_disp)

    marginal = NegBin({"mean": mean_formula, "dispersion": dispersion_formula}, device=device)
    copula = StandardCopula(copula_formula, estimator=estimator, top_k=top_k)
    return spatial_copula(
        marginal,
        terms,
        copula=copula,
        spatial_cols=spatial_cols,
        cap_at_observed_max=cap_at_observed_max,
    )


def PenalizedNegBinCopula(
    mean_formula: str = "~ 1",
    dispersion_formula: str = "~ 1",
    copula_formula: str = "~ 1",
    *,
    penalties: Sequence[CoefficientPenalty] = (),
    cap_at_observed_max: bool = True,
    estimator=None,
    top_k: Optional[int] = None,
    device=None,
) -> SCD3Simulator:
    """A regularized negative binomial simulator

    The simulator is well-suited to count data with many covariates relative to
    the number of cells.

    Parameters
    ----------
    mean_formula, dispersion_formula, copula_formula : str
    penalties : sequence of CoefficientPenalty
        See :mod:`scdesigner.distributions.penalties`.
    cap_at_observed_max : bool
        Cap samples with :class:`~scdesigner.simulators.output.ObservedMaxCap`.
    estimator, top_k
        Passed to :class:`StandardCopula`.
    device : torch.device or str, optional
        Marginal device.

    Examples
    --------
    >>> from scdesigner.distributions.penalties import CoefficientPenalty, QuadraticPenalty
    >>> penalty = CoefficientPenalty("mean", ["x"], [1.0], QuadraticPenalty(lam=0.1))
    >>> sim = PenalizedNegBinCopula("~ x", penalties=[penalty])
    >>> sim.fit(adata)  # doctest: +SKIP
    """
    marginal = PenalizedNegBin(
        {"mean": mean_formula, "dispersion": dispersion_formula},
        penalties=penalties,
        device=device,
    )
    copula = StandardCopula(copula_formula, estimator=estimator, top_k=top_k)
    return SCD3Simulator(
        marginal,
        copula,
        output_transform=ObservedMaxCap() if cap_at_observed_max else None,
    )
