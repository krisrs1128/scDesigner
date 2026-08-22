from ..base.copula import Copula, CovarianceStructure
from ..covariance import as_covariance_estimator
from ..data.formula import standardize_formula
from ..utils.kwargs import DEFAULT_ALLOWED_KWARGS, _filter_kwargs
from .gaussian_copula_helpers import (
    batch_log_likelihood,
    covariance_to_correlation,
    factorize_correlation,
    group_log_likelihood,
    sample_pseudo_obs,
)
from anndata import AnnData
from scipy.stats import norm
from tqdm import tqdm
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import warnings


class StandardCopula(Copula):
    """
    Gaussian copula with modular covariance estimators.

    The copula estimates the correlation matrix for gaussianized data. We
    support alternative (e.g., regularized) covariance estimators through
    :class:`~scdesigner.covariance.CovarianceEstimator`.

    Parameters
    ----------
    formula : str or dict, optional
        How the copula depends on experimental or biological conditions.
        Defaults to ``"~ 1"``, a single group.
    estimator : CovarianceEstimator, str, or None, optional
        An estimator, a registered name such as ``"ledoit_wolf"`` or ``"oas"``,
        or ``None`` for the ordinary sample covariance.
    top_k : int, optional
        Model only the ``top_k`` most expressed genes jointly. Rest treated as
        independent. A simple form of regularization.

    Attributes
    ----------
    loader : torch.utils.data.DataLoader
        Yields the batches that are used to estimate the covariance.
    n_outcomes : int
        Number of features (genes) modeled.
    parameters : Dict[str, CovarianceStructure]
        Fitted correlation structure per group.
    estimators_ : Dict[str, CovarianceEstimator]
        Fitted estimator per group, kept for diagnostics.
    groups : list
        Group labels, one per column of the ``"group"`` design.
    n_groups : int
        Number of groups.

    Examples
    --------
    >>> from scdesigner.copulas import StandardCopula
    >>> from scdesigner.covariance import LedoitWolf
    >>>
    >>> copula = StandardCopula("~ cell_type", estimator=LedoitWolf())
    >>> copula.formula
    {'group': '~ cell_type'}

    See Also
    --------
    :class:`~scdesigner.covariance.CovarianceEstimator`
    """

    def __init__(
        self,
        formula: Union[str, dict] = "~ 1",
        estimator=None,
        top_k: Optional[int] = None,
    ):
        formula = standardize_formula(formula, allowed_keys=["group"])
        super().__init__(formula)

        # convert string name to an estimator object
        as_covariance_estimator(estimator)
        self.estimator = estimator
        self.top_k = top_k

        self.groups = None
        self.copula_likelihood = 0
        self.estimators_ = {}

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup_data(self, adata: AnnData, marginal_formula: Dict[str, str], **kwargs):
        """
        Build the loader and the group design.

        Parameters
        ----------
        adata : AnnData
            Cells in rows, features in columns.
        marginal_formula : dict of {str: str}
            Formulas used by the marginal models, we merge it with the copula's.
        **kwargs
            Passed to :func:`~scdesigner.data.adata_loader`.
        """
        data_kwargs = _filter_kwargs(kwargs, DEFAULT_ALLOWED_KWARGS["data"])
        super().setup_data(adata, marginal_formula, **data_kwargs)

        self.groups = self.loader.dataset.predictor_names["group"]
        self.n_groups = len(self.groups)
        self._group_col = {g: i for i, g in enumerate(self.groups)}

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, uniformizer: Callable):
        r"""
        Estimate a correlation matrix for each group.

        The data are first gaussianized using the estimated marginals then we
        stream across batches to estimate the covariance.

        The fit is controlled entirely by the copula's construction: ``estimator``
        and ``top_k`` are set in :meth:`__init__`, and the batching is set in
        :meth:`setup_data`.

        Parameters
        ----------
        uniformizer : callable
            ``uniformizer(y, x_dict) -> np.ndarray`` mapping expression data to
            uniform \([0, 1]\) values.
        """
        prototype = as_covariance_estimator(self.estimator)
        modeled_indices, remaining_indices = self._gene_partitions()

        self.copula_likelihood = 0
        self.estimators_ = self._fit_estimators(prototype, uniformizer, modeled_indices)
        self.parameters = self._build_covariances(modeled_indices, remaining_indices)

    def _gene_partitions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split genes into those modeled jointly and those left independent.

        Genes are ranked by total expression and the top_k are modeled jointly.
        The rest are modeled independently.
        """
        top_k = self.top_k
        if top_k is None:
            return np.arange(self.n_outcomes), np.array([], dtype=int)

        gene_total_expression = np.asarray(self.adata.X.sum(axis=0)).flatten()
        sorted_indices = np.argsort(gene_total_expression)
        return np.sort(sorted_indices[-top_k:]), np.sort(sorted_indices[:-top_k])

    def _fit_estimators(
        self, prototype, uniformizer: Callable, modeled_indices: np.ndarray
    ) -> Dict[Union[str, int], object]:
        """
        Estimate covariance matrices for each group
        """
        n_modeled = len(modeled_indices)
        estimators = {g: prototype.clone() for g in self.groups}
        for estimator in estimators.values():
            estimator.start(n_modeled)

        desc = (
            "Estimating top-k copula correlation"
            if n_modeled < self.n_outcomes
            else "Estimating copula correlation"
        )
        for y, x_dict in tqdm(self.loader, desc=desc):
            memberships = x_dict["group"].cpu().numpy()
            z = np.asarray(norm.ppf(uniformizer(y, x_dict)), dtype=float)

            for g, estimator in estimators.items():
                mask = memberships[:, self._group_col[g]] == 1
                if np.any(mask):
                    estimator.update(z[mask][:, modeled_indices])

        for estimator in estimators.values():
            estimator.finalize()
        return estimators

    def _build_covariances(
        self, modeled_indices: np.ndarray, remaining_indices: np.ndarray
    ) -> Dict[Union[str, int], CovarianceStructure]:
        """Turn each fitted estimate into a correlation structure for sampling."""
        covariances = {}
        has_remainder = len(remaining_indices) > 0
        remaining_var = np.ones(len(remaining_indices)) if has_remainder else None

        for g, estimator in self.estimators_.items():
            corr = covariance_to_correlation(estimator.covariance_)
            corr, factor = factorize_correlation(corr, g, jitter=estimator.regularized)
            self._add_loglikelihood(factor, estimator)
            estimator.release()

            covariances[g] = CovarianceStructure(
                cov=corr,
                modeled_names=self.adata.var_names[modeled_indices],
                modeled_indices=modeled_indices,
                remaining_var=remaining_var,
                remaining_indices=remaining_indices if has_remainder else None,
                remaining_names=(
                    self.adata.var_names[remaining_indices] if has_remainder else None
                ),
            )
        return covariances

    def _add_loglikelihood(self, factor, estimator) -> None:
        """
        Add one group's contribution to the log-likelihood.

        This will update the likelihood for any method that has a non-null
        copula_likelihood and running second_moment calculation. If these aren't
        present, you can run ``complexity(adata=...)`` to recompute it in a
        second pass.
        """
        if self.copula_likelihood is None:
            return

        second_moment = estimator.second_moment_
        if second_moment is None:
            self.copula_likelihood = None
            warnings.warn(
                f"""{type(estimator).__name__} does not return copula likelihoods
                in the first pass. Call complexity(adata=...) to recompute it
                from the data."""
            )
            return

        self.copula_likelihood += group_log_likelihood(
            factor, second_moment, estimator.n_samples_
        )

    # ------------------------------------------------------------------
    # sampling and evaluation
    # ------------------------------------------------------------------
    def _group_indices(self, x_dict: Dict) -> Dict[Union[str, int], np.ndarray]:
        """Row indices belonging to each group, for one batch."""
        memberships = x_dict["group"].cpu().numpy()
        return {
            g: np.where(memberships[:, self._group_col[g]] == 1)[0] for g in self.groups
        }

    def pseudo_obs(self, x_dict: Dict):
        """
        Draw dependent uniform pseudo-observations.

        Parameters
        ----------
        x_dict : dict
            Batch covariates, including the ``"group"`` indicator matrix.

        Returns
        -------
        np.ndarray
            Uniform values, shape ``(n_cells, n_genes)``.
        """
        group_ix = self._group_indices(x_dict)
        n_cells = len(x_dict["group"])

        u = np.zeros((n_cells, self.n_outcomes))
        for group, cov_struct in self.parameters.items():
            indices = group_ix[group]
            if len(indices) > 0:
                u[indices] = sample_pseudo_obs(len(indices), cov_struct)
        return u

    def likelihood(self, uniformizer: Callable, batch):
        """
        Per-cell copula log-likelihood for one batch.

        Parameters
        ----------
        uniformizer : callable
            Maps expression data to uniform values.
        batch : tuple of (torch.Tensor, dict)
            Expression tensor and covariates, as yielded by the loader.

        Returns
        -------
        np.ndarray
            Log-likelihood per cell.
        """
        y, x_dict = batch
        z = norm.ppf(uniformizer(y, x_dict))
        group_ix = self._group_indices(x_dict)

        ll = np.zeros(len(z))
        for group, cov_struct in self.parameters.items():
            indices = group_ix[group]
            if len(indices) > 0:
                ll[indices] = batch_log_likelihood(z[indices], cov_struct)
        return ll

    def num_params(self, **kwargs):
        """
        Number of correlation parameters, summed over groups.
        """
        return int(
            sum(
                estimator.num_params(estimator.n_features_)
                for estimator in self.estimators_.values()
                if estimator.n_samples_ > 0
            )
        )

    @property
    def diagnostics(self) -> Dict:
        """Per-group estimator diagnostics, such as the fitted shrinkage intensity."""
        return {g: e.diagnostics() for g, e in self.estimators_.items()}
