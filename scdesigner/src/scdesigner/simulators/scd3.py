from ..base.copula import Copula
from ..data.loader import obs_loader, adata_loader
from ..data import basis_formula, GPBasis, TPSBasis
from ..base.marginal import Marginal
from ..base.simulator import Simulator
from anndata import AnnData
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch
from ..distributions import (
    NegBin,
    NegBinIRLS,
    PenalizedNegBin,
    ZeroInflatedNegBin,
    Gaussian,
    Poisson,
    ZeroInflatedPoisson,
    Bernoulli,
)
from ..copulas import StandardCopula
from typing import Optional
from abc import ABC


class SCD3Simulator(Simulator, ABC):
    """High-level simulation wrapper combining a marginal model and a copula.

    The :class:`SCD3Simulator` class coordinates fitting of a marginal model
    (e.g. negative binomial, zero-inflated negative binomial) together with a
    dependence structure specified by a copula (e.g. :class:`StandardCopula`).
    Subclasses provide concrete combinations of marginal and copula models
    tailored to common use cases.

    Parameters
    ----------
    marginal : Marginal
        Fitted or unfitted marginal model describing the distribution of each
        feature (e.g. gene) conditional on covariates.
    copula : Copula
        Copula object that captures dependence between features and shares
        the same covariate structure as the marginal model.

    Attributes
    ----------
    marginal : Marginal
        The marginal model instance used for fitting and simulation.
    copula : Copula
        The copula instance used to model dependence between features.
    template : AnnData or None
        Reference dataset used to define the observed covariate space and
        feature set for simulation. Set during :meth:`fit`.
    parameters : dict or None
        Dictionary containing fitted parameters for both the marginal and
        copula components after :meth:`fit` has been called.

    Examples
    --------
    The abstract :class:`SCD3Simulator` is not used directly. Instead, use one
    of its concrete subclasses, e.g. :class:`NegBinCopula`::

        >>> import scanpy as sc
        >>> from scdesigner.simulators.scd3 import NegBinCopula
        >>>
        >>> adata = sc.datasets.pbmc3k()[:, :100].copy()
        >>>
        >>> # Mean expression depends on group; copula uses the same group structure
        >>> sim = NegBinCopula(mean_formula="~ 1", dispersion_formula="~ 1",
        ...                    copula_formula="~ 1")
        >>> sim.fit(adata, batch_size=256, max_epochs=10) # doctest: +SKIP
        >>>
        >>> # Generate synthetic data with the same obs covariates
        >>> synthetic = sim.sample(batch_size=512) # doctest: +SKIP
        >>> synthetic.X.shape == adata.shape # doctest: +SKIP
        True
        >>>
        >>> # Compute model complexity via AIC/BIC of the copula component
        >>> metrics = sim.complexity() # doctest: +SKIP
        >>> sorted(metrics.keys()) # doctest: +SKIP
        ['aic', 'bic'] # doctest: +SKIP
    """

    def __init__(self, marginal: Marginal, copula: Copula):
        self.marginal = marginal
        self.copula = copula
        self.template = None
        self.parameters = None

    def fit(self, adata: AnnData, **kwargs):
        """Fit marginal and copula components to an AnnData object.

        Parameters
        ----------
        adata : AnnData
            Input dataset with cells in rows and features (e.g. genes) in
            columns. Both the marginal and copula components are fitted to
            this data.
        **kwargs
            Additional keyword arguments forwarded to the marginal and copula
            fit routines (e.g. ``batch_size``, optimization settings).

        Notes
        -----
        This method sets the :attr:`template` attribute to ``adata`` and
        stores fitted parameters in :attr:`parameters`.
        """
        self.template = adata
        self.marginal.setup_data(adata, **kwargs)
        self.marginal.setup_optimizer(**kwargs)
        self.marginal.fit(**kwargs)

        # copula simulator
        self.copula.setup_data(adata, self.marginal.formula, **kwargs)
        self.copula.fit(self.marginal.uniformize, **kwargs)
        self.parameters = {
            "marginal": self.marginal.parameters,
            "copula": self.copula.parameters,
        }

    def predict(self, obs=None, batch_size: int = 1000, **kwargs):
        """Predict marginal parameters for given covariates.

        Parameters
        ----------
        obs : pandas.DataFrame or None, optional
            Observation-level covariate table. If ``None``, use
            ``self.template.obs`` from the dataset provided to :meth:`fit`.
        batch_size : int, optional
            Number of observations per mini-batch used during prediction.
        **kwargs
            Additional keyword arguments passed to :func:`obs_loader`.

        Returns
        -------
        dict
            Dictionary mapping parameter names (e.g. ``"mean"``,
            ``"dispersion"``) to NumPy arrays of shape ``(n_cells, n_genes)``
            containing the predicted marginal parameters.
        """
        # prepare an internal data loader for this obs
        if obs is None:
            obs = self.template.obs
        loader = obs_loader(obs, self.marginal.formula, batch_size=batch_size, **kwargs)

        # get predictions across batches
        local_parameters = []
        for _, x_dict in loader:
            l = self.marginal.predict(x_dict)
            local_parameters.append(l)

        # convert to a merged dictionary
        keys = list(local_parameters[0].keys())
        return {
            k: torch.cat([d[k] for d in local_parameters]).detach().cpu().numpy()
            for k in keys
        }

    def sample(self, obs=None, batch_size: int = 1000, **kwargs):
        """Generate synthetic observations from the fitted model.

        Parameters
        ----------
        obs : pandas.DataFrame or None, optional
            Observation-level covariate table defining the covariate space
            for simulation. If ``None``, use ``self.template.obs``.
        batch_size : int, optional
            Number of observations per mini-batch used during sampling.
        **kwargs
            Additional keyword arguments passed to :func:`obs_loader`.

        Returns
        -------
        AnnData
            An :class:`AnnData` object with simulated counts in ``.X`` and
            ``obs`` equal to the provided covariate table.
        """
        if obs is None:
            obs = self.template.obs
        loader = obs_loader(
            obs,
            self.copula.formula | self.marginal.formula,
            batch_size=batch_size,
            **kwargs,
        )

        # get samples across batches
        samples = []
        for _, x_dict in loader:
            u = self.copula.pseudo_obs(x_dict)
            u = torch.from_numpy(u)
            samples.append(self.marginal.invert(u, x_dict))
        samples = torch.cat(samples).detach().cpu().numpy()
        return AnnData(X=samples, obs=obs, var=self.template.var)

    def complexity(self, adata: AnnData = None, **kwargs):
        """Compute model complexity metrics (AIC, BIC) for the copula component.

        Parameters
        ----------
        adata : AnnData or None, optional
            Dataset to evaluate the copula log-likelihood on. If ``None``,
            use the template dataset stored during :meth:`fit`.
        **kwargs
            Additional keyword arguments passed to :func:`adata_loader`.

        Returns
        -------
        dict
            Dictionary with keys ``"aic"`` and ``"bic"`` computed from the
            copula log-likelihood and :meth:`copula.num_params`.
        """
        use_template_ll = False
        if adata is None:
            use_template_ll = True
            adata = self.template

        N, marginal_ll, copula_ll = 0, 0, 0
        loader = adata_loader(adata, self.marginal.formula | self.copula.formula, **kwargs)
        for batch in tqdm(loader, desc="Computing log-likelihood..."):
            with torch.no_grad():
                marginal_ll += self.marginal.likelihood(batch).sum()
            if not use_template_ll:
                copula_ll += self.copula.likelihood(self.marginal.uniformize, batch).sum()
            N += len(batch[0])

        if use_template_ll: copula_ll = self.copula.copula_likelihood

        return {
            "marginal_aic": -2 * marginal_ll.item() + 2 * self.marginal.num_params(),
            "marginal_bic": -2 * marginal_ll.item() + np.log(N) * self.marginal.num_params(),
            "copula_aic": -2 * copula_ll + 2 * self.copula.num_params(),
            "copula_bic": -2 * copula_ll + np.log(N) * self.copula.num_params()
        }


################################################################################
## SCD3 instances
################################################################################


class NegBinCopula(SCD3Simulator):
    """Simulator using negative binomial marginals with a Gaussian copula.

    Parameters
    ----------
    mean_formula : str or None, optional
        Model formula for the mean parameter of the negative binomial
        marginal (e.g. ``"~ 1"`` or ``"~ group"``). If ``None``, a
        default constant-mean formula is used.
    dispersion_formula : str or None, optional
        Model formula for the dispersion parameter of the negative
        binomial marginal. If ``None``, a default constant-dispersion
        formula is used.
    copula_formula : str or None, optional
        Copula formula describing how copula depends on experimental
        or biological conditions (e.g. ``"~ group"``).If ``None``,
        a default intercept-only formula is used.

    See Also
    --------
    :class:`SCD3Simulator`
    :class:`NegBin`
    :class:`StandardCopula`
    """

    def __init__(
        self,
        mean_formula: Optional[str] = "~ 1",
        dispersion_formula: Optional[str] = "~ 1",
        copula_formula: Optional[str] = "~ 1",
    ) -> None:
        marginal = NegBin({"mean": mean_formula, "dispersion": dispersion_formula})
        covariance = StandardCopula(copula_formula)
        super().__init__(marginal, covariance)


class ZeroInflatedNegBinCopula(SCD3Simulator):
    """Simulator using zero-inflated negative binomial marginals with
    a Gaussian copula.

    Parameters
    ----------
    mean_formula : str or None, optional
        Model formula for the mean parameter of the zero-inflated
        negative binomial marginal (e.g. ``"~ 1"`` or ``"~ group"``).
        If ``None``, a default constant-mean formula is used.
    dispersion_formula : str or None, optional
        Model formula for the dispersion parameter of the zero-inflated
        negative binomial marginal. If ``None``, a default
        constant-dispersion formula is used.
    zero_inflation_formula : str or None, optional
        Model formula for the zero-inflation parameter of the zero-inflated
        negative binomial marginal. If ``None``, a default
        constant-zero-inflation formula is used.
    copula_formula : str or None, optional
        Copula formula describing how copula depends on experimental or
        biological conditions (e.g. ``"~ group"``). If ``None``, a default
        intercept-only formula is used.

    See Also
    --------
    :class:`SCD3Simulator`
    :class:`ZeroInflatedNegBin`
    :class:`StandardCopula`
    """

    def __init__(
        self,
        mean_formula: Optional[str] = "~ 1",
        dispersion_formula: Optional[str] = "~ 1",
        zero_inflation_formula: Optional[str] = "~ 1",
        copula_formula: Optional[str] = "~ 1",
    ) -> None:
        marginal = ZeroInflatedNegBin(
            {
                "mean": mean_formula,
                "dispersion": dispersion_formula,
                "zero_inflation": zero_inflation_formula,
            }
        )
        covariance = StandardCopula(copula_formula)
        super().__init__(marginal, covariance)


class BernoulliCopula(SCD3Simulator):
    """Simulator using Bernoulli marginals with a Gaussian copula.

    Parameters
    ----------
    mean_formula : str or None, optional
        Model formula for the mean parameter of the Bernoulli marginal
        (e.g. ``"~ 1"`` or ``"~ group"``). If ``None``, a default
        constant-mean formula is used.
    copula_formula : str or None, optional
        Copula formula describing how copula depends on experimental or
        biological conditions (e.g. ``"~ group"``). If ``None``, a default
        intercept-only formula is used.

    See Also
    --------
    :class:`SCD3Simulator`
    :class:`Bernoulli`
    :class:`StandardCopula`
    """

    def __init__(
        self, mean_formula: Optional[str] = "~ 1", copula_formula: Optional[str] = "~ 1"
    ) -> None:
        marginal = Bernoulli({"mean": mean_formula})
        covariance = StandardCopula(copula_formula)
        super().__init__(marginal, covariance)


class GaussianCopula(SCD3Simulator):
    """Simulator using Gaussian marginals with a Gaussian copula.

    Parameters
    ----------
    mean_formula : str or None, optional
        Model formula for the mean parameter of the Gaussian marginal
        (e.g. ``"~ 1"`` or ``"~ group"``). If ``None``, a default
        constant-mean formula is used.
    sdev_formula : str or None, optional
        Model formula for the standard deviation parameter of the Gaussian marginal.
        If ``None``, a default constant-standard deviation formula is used.
    copula_formula : str or None, optional
        Copula formula describing how copula depends on experimental or
        biological conditions (e.g. ``"~ group"``). If ``None``, a default
        intercept-only formula is used.

    See Also
    --------
    :class:`SCD3Simulator`
    :class:`Gaussian`
    :class:`StandardCopula`
    """

    def __init__(
        self,
        mean_formula: Optional[str] = "~ 1",
        sdev_formula: Optional[str] = "~ 1",
        copula_formula: Optional[str] = "~ 1",
    ) -> None:
        marginal = Gaussian({"mean": mean_formula, "sdev": sdev_formula})
        covariance = StandardCopula(copula_formula)
        super().__init__(marginal, covariance)


class PoissonCopula(SCD3Simulator):
    """Simulator using Poisson marginals with a Gaussian copula.

    Parameters
    ----------
    mean_formula : str or None, optional
        Model formula for the mean parameter of the Poisson marginal
        (e.g. ``"~ 1"`` or ``"~ group"``). If ``None``, a default
        constant-mean formula is used.
    copula_formula : str or None, optional
        Copula formula describing how copula depends on experimental or
        biological conditions (e.g. ``"~ group"``). If ``None``, a default
        intercept-only formula is used.

    See Also
    --------
    :class:`SCD3Simulator`
    :class:`Poisson`
    :class:`StandardCopula`
    """

    def __init__(
        self, mean_formula: Optional[str] = "~ 1", copula_formula: Optional[str] = "~ 1"
    ) -> None:
        marginal = Poisson({"mean": mean_formula})
        covariance = StandardCopula(copula_formula)
        super().__init__(marginal, covariance)


class ZeroInflatedPoissonCopula(SCD3Simulator):
    """Simulator using zero-inflated Poisson marginals with a Gaussian copula.

    Parameters
    ----------
    mean_formula : str or None, optional
        Model formula for the mean parameter of the zero-inflated Poisson marginal
        (e.g. ``"~ 1"`` or ``"~ group"``). If ``None``, a default
        constant-mean formula is used.
    zero_inflation_formula : str or None, optional
        Model formula for the zero-inflation parameter of the zero-inflated Poisson
        marginal. If ``None``, a default constant-zero-inflation formula is used.
    copula_formula : str or None, optional
        Copula formula describing how copula depends on experimental or
        biological conditions (e.g. ``"~ group"``). If ``None``, a default
        intercept-only formula is used.

    See Also
    --------
    :class:`SCD3Simulator`
    :class:`ZeroInflatedPoisson`
    :class:`StandardCopula`
    """

    def __init__(
        self,
        mean_formula: Optional[str] = "~ 1",
        zero_inflation_formula: Optional[str] = "~ 1",
        copula_formula: Optional[str] = "~ 1",
    ) -> None:
        marginal = ZeroInflatedPoisson(
            {"mean": mean_formula, "zero_inflation": zero_inflation_formula}
        )
        covariance = StandardCopula(copula_formula)
        super().__init__(marginal, covariance)

class NegBinIRLSCopula(SCD3Simulator):
    """Simulator using negative binomial marginals with a Gaussian copula.

    Parameters
    ----------
    mean_formula : str or None, optional
        Model formula for the mean parameter of the negative binomial
        marginal (e.g. ``"~ 1"`` or ``"~ group"``). If ``None``, a
        default constant-mean formula is used.
    dispersion_formula : str or None, optional
        Model formula for the dispersion parameter of the negative
        binomial marginal. If ``None``, a default constant-dispersion
        formula is used.
    copula_formula : str or None, optional
        Copula formula describing how copula depends on experimental
        or biological conditions (e.g. ``"~ group"``).If ``None``,
        a default intercept-only formula is used.

    See Also
    --------
    :class:`SCD3Simulator`
    :class:`NegBin`
    :class:`StandardCopula`
    """

    def __init__(
        self,
        mean_formula: Optional[str] = "~ 1",
        dispersion_formula: Optional[str] = "~ 1",
        copula_formula: Optional[str] = "~ 1"
    ) -> None:
        marginal = NegBinIRLS({"mean": mean_formula, "dispersion": dispersion_formula})
        covariance = StandardCopula(copula_formula)
        super().__init__(marginal, covariance)

    def fit(self, adata: AnnData, batch_size: int = 8224, device="cpu", **kwargs):
        super().fit(adata, batch_size=batch_size, device=device, **kwargs)

    def sample(self, obs=None, batch_size: int = 8224, **kwargs):
        return super().sample(obs, batch_size, device="cpu", **kwargs)

    def predict(self, obs=None, batch_size: int = 8224, **kwargs):
        return super().predict(obs, batch_size, device="cpu", **kwargs)


class PenalizedNegBinCopula(SCD3Simulator):
    """NB copula simulator with smoothness penalties on spatial basis coefficients
    for both mean and dispersion models, plus per-gene count capping.

    Parameters
    ----------
    mean_formula, dispersion_formula, copula_formula : str
        As in NegBinCopula.
    mean_penalty_diag : np.ndarray or None
        Diagonal of the penalty matrix for mean spatial coefficients.
    disp_penalty_diag : np.ndarray or None
        Diagonal of the penalty matrix for dispersion spatial coefficients.
    mean_basis_cols : list of str
        Names of the penalised basis columns in the mean design matrix.
    disp_basis_cols : list of str
        Names of the penalised basis columns in the dispersion design matrix.
    lam : float
        Base penalty strength for mean model.
    lam_disp : float
        Penalty strength for dispersion model.
    gene_means : np.ndarray or None
        Per-gene mean expression for adaptive penalty scaling (1/mean_j).
    cap_at_observed_max : bool
        If True, clip simulated counts to the per-gene observed maximum.
    """

    def __init__(
        self,
        mean_formula="~ 1",
        dispersion_formula="~ 1",
        copula_formula="~ 1",
        mean_penalty_diag=None,
        disp_penalty_diag=None,
        mean_basis_cols=None,
        disp_basis_cols=None,
        lam=1.0,
        lam_disp=1.0,
        gene_means=None,
        cap_at_observed_max=True,
    ):
        self._lam = lam
        self._lam_disp = lam_disp
        self._gene_means = gene_means
        self._cap_at_observed_max = cap_at_observed_max
        self._gene_max = None  # set during fit
        marginal = PenalizedNegBin(
            {"mean": mean_formula, "dispersion": dispersion_formula},
            mean_penalty_diag=mean_penalty_diag,
            disp_penalty_diag=disp_penalty_diag,
            mean_basis_cols=mean_basis_cols,
            disp_basis_cols=disp_basis_cols,
            lam=lam,
            lam_disp=lam_disp,
            gene_means=gene_means,
        )
        copula = StandardCopula(copula_formula)
        super().__init__(marginal, copula)

    def fit(self, adata, **kwargs):
        # Store per-gene max before fitting
        if self._cap_at_observed_max:
            X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
            self._gene_max = X.max(axis=0).flatten()
        return super().fit(adata, **kwargs)

    def sample(self, **kwargs):
        result = super().sample(**kwargs)
        if self._cap_at_observed_max and self._gene_max is not None:
            X = result.X
            result.X = np.minimum(X, self._gene_max[None, :])
        return result


class SpatialNegBinCopula(SCD3Simulator):
    """NB copula simulator for spatial expression.

    Builds two stateful :class:`SpatialBasis` objects (mean, dispersion) at
    fit time and reuses them at predict / sample on fresh coordinates.
    For custom designs or non-spatial penalties, use
    :class:`PenalizedNegBinCopula` directly.

    Parameters
    ----------
    mean_df, disp_df : int
        Basis rank for the mean and dispersion models.
    basis : {"gp", "tps"}
        Basis class (:class:`GPBasis` or :class:`TPSBasis`).
    standardize : bool
        Rescale basis columns by training std.
    mean_extra_terms, disp_extra_terms : list of str or None
        Extra additive covariates appended to each formula. Drops the intercept
        when set.
    copula_formula : str
        Gaussian copula formula.
    lam, lam_disp : float
        Penalty strengths for mean and dispersion coefficients.
    cap_at_observed_max : bool
        Clip simulated counts to per-gene observed max.
    spatial_cols : tuple of str
        Coordinate column names in ``adata.obs``.
    **basis_kwargs
        Forwarded to the chosen :class:`SpatialBasis` subclass
        (e.g. ``n_landmarks``, ``length_scale``).
    """

    _BASIS_CLASSES = {"tps": TPSBasis, "gp": GPBasis}

    def __init__(
        self,
        mean_df: int = 15,
        disp_df: int = 5,
        basis: str = "gp",
        standardize: bool = True,
        mean_extra_terms: Optional[list] = None,
        disp_extra_terms: Optional[list] = None,
        copula_formula: str = "~ 1",
        lam: float = 1.0,
        lam_disp: float = 1.0,
        cap_at_observed_max: bool = True,
        spatial_cols: tuple = ("spatial1", "spatial2"),
        **basis_kwargs,
    ):
        if basis not in self._BASIS_CLASSES:
            raise ValueError(
                f"Unknown basis {basis!r}; expected one of "
                f"{sorted(self._BASIS_CLASSES)}"
            )
        self.marginal = self.copula = self.template = None
        self.parameters = self._gene_max = None
        self._mean_basis = None
        self._disp_basis = None
        self._mean_cols = None
        self._disp_cols = None
        self._config = {k: v for k, v in locals().items() if k != "self"}

    def _make_basis(self, df):
        cfg = self._config
        cls = self._BASIS_CLASSES[cfg["basis"]]
        return cls(df=df, standardize=cfg["standardize"], **cfg["basis_kwargs"])

    def _coords_from_obs(self, obs):
        col1, col2 = self._config["spatial_cols"]
        return np.column_stack([obs[col1].values, obs[col2].values])

    def _augment_obs(self, obs, fit):
        """Return a copy of ``obs`` with ``sp_mean_*`` and ``sp_disp_*`` columns.

        ``fit=True`` instantiates and fits the bases on these coordinates;
        ``fit=False`` reuses the stored bases via ``transform``.
        """
        coords = self._coords_from_obs(obs)
        if fit:
            self._mean_basis = self._make_basis(self._config["mean_df"])
            self._disp_basis = self._make_basis(self._config["disp_df"])
            mean_B = self._mean_basis.fit_transform(coords)
            disp_B = self._disp_basis.fit_transform(coords)
            self._mean_cols = [f"sp_mean_{i}" for i in range(mean_B.shape[1])]
            self._disp_cols = [f"sp_disp_{i}" for i in range(disp_B.shape[1])]
        else:
            mean_B = self._mean_basis.transform(coords)
            disp_B = self._disp_basis.transform(coords)

        # add these basis coolumns to the obs data
        obs = obs.copy()
        for i, c in enumerate(self._mean_cols):
            obs[c] = mean_B[:, i]
        for i, c in enumerate(self._disp_cols):
            obs[c] = disp_B[:, i]
        return obs

    def fit(self, adata: AnnData, **kwargs):
        """Fit spatial bases, marginal, and copula.

        The bases are stored so :meth:`predict` / :meth:`sample` can project
        fresh coordinates with the same landmarks and SVD. The user's
        ``adata`` is not mutated: :attr:`template` is a lightweight wrapper
        sharing ``X`` and ``var`` with an augmented ``obs``.
        """
        cfg = self._config
        new_obs = self._augment_obs(adata.obs, fit=True)
        adata = AnnData(X=adata.X, obs=new_obs, var=adata.var)

        # preparatory data and formula context
        X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
        gene_means = X.mean(axis=0).flatten()
        if cfg["cap_at_observed_max"]:
            self._gene_max = X.max(axis=0).flatten()

        mean_formula = basis_formula(self._mean_cols, cfg["mean_extra_terms"])
        disp_formula = basis_formula(self._disp_cols, cfg["disp_extra_terms"])

        # call a negbin simulator with this context
        self.marginal = PenalizedNegBin(
            {"mean": mean_formula, "dispersion": disp_formula},
            mean_penalty_diag=self._mean_basis.penalty_diag,
            disp_penalty_diag=self._disp_basis.penalty_diag,
            mean_basis_cols=self._mean_cols,
            disp_basis_cols=self._disp_cols,
            lam=cfg["lam"],
            lam_disp=cfg["lam_disp"],
            gene_means=gene_means,
        )
        self.copula = StandardCopula(cfg["copula_formula"])
        SCD3Simulator.fit(self, adata, **kwargs)

    def complexity(self, adata: AnnData = None, **kwargs):
        if adata is not None:
            adata = AnnData(
                X=adata.X,
                obs=self._augment_obs(adata.obs, fit=False),
                var=adata.var,
            )
        return super().complexity(adata, **kwargs)

    def predict(self, obs=None, **kwargs):
        # need to recreate the spatial basis before predicting
        if obs is not None:
            obs = self._augment_obs(obs, fit=False)

        return super().predict(obs=obs, **kwargs)

    def sample(self, obs=None, batch_size: int = 1000, **kwargs):
        # need to recreate the spatial basis before sampling
        if obs is not None:
            obs = self._augment_obs(obs, fit=False)

        # this cap helps avoid outliers created by poor dispersion fits
        result = super().sample(obs=obs, batch_size=batch_size, **kwargs)
        if self._config["cap_at_observed_max"] and self._gene_max is not None:
            result.X = np.minimum(result.X, self._gene_max[None, :])
        return result