from ..base.copula import Copula
from ..data.loader import obs_loader, adata_loader
from ..base.marginal import Marginal
from ..base.simulator import Simulator
from anndata import AnnData
from tqdm import tqdm
import torch
import numpy as np
from ..distributions import (
    NegBin,
    NegBinIRLS,
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
        return AnnData(X=samples, obs=obs)

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
        if adata is None:
            adata = self.template

        N, ll = 0, 0
        loader = adata_loader(
            adata, self.marginal.formula | self.copula.formula, **kwargs
        )
        for batch in tqdm(loader, desc="Computing log-likelihood..."):
            ll += self.copula.likelihood(self.marginal.uniformize, batch).sum()
            N += len(batch[0])

        return {
            "aic": -2 * ll + 2 * self.copula.num_params(),
            "bic": -2 * ll + np.log(N) * self.copula.num_params(),
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
        mean_formula: Optional[str] = None,
        dispersion_formula: Optional[str] = None,
        copula_formula: Optional[str] = None,
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
        mean_formula: Optional[str] = None,
        dispersion_formula: Optional[str] = None,
        zero_inflation_formula: Optional[str] = None,
        copula_formula: Optional[str] = None,
    ) -> None:
        marginal = ZeroInflatedNegBin(
            {
                "mean": mean_formula,
                "dispersion": dispersion_formula,
                "zero_inflation_formula": zero_inflation_formula,
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
        self, mean_formula: Optional[str] = None, copula_formula: Optional[str] = None
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
        mean_formula: Optional[str] = None,
        sdev_formula: Optional[str] = None,
        copula_formula: Optional[str] = None,
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
        self, mean_formula: Optional[str] = None, copula_formula: Optional[str] = None
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
        mean_formula: Optional[str] = None,
        zero_inflation_formula: Optional[str] = None,
        copula_formula: Optional[str] = None,
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
        mean_formula: Optional[str] = None,
        dispersion_formula: Optional[str] = None,
        copula_formula: Optional[str] = None
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