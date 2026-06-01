from ..base.copula import Copula
from ..data.formula import standardize_formula
from ..utils.kwargs import DEFAULT_ALLOWED_KWARGS, _filter_kwargs
from anndata import AnnData
from scipy.stats import norm, multivariate_normal
from tqdm import tqdm
from typing import Dict, Union, Callable, Tuple
import numpy as np
import torch
from ..base.copula import CovarianceStructure
import warnings


class StandardCopula(Copula):
    """
    Gaussian copula model with optional group-specific correlation structures.

    This implementation estimates a multivariate normal dependence structure
    on latent Gaussian variables. Optionally, different covariance
    matrices can be estimated for categorical groups (for example, cell
    types or experimental conditions).

    Parameters
    ----------
    formula : str or dict, optional
        A formula describing how the copula depends on experimental or
        biological conditions. The formula is standardized to ensure that
        a ``"group"`` term is always present. By default ``"~ 1"``.

    Attributes
    ----------
    loader : torch.utils.data.DataLoader
        A data loader object is used to estimate the covariance one batch at a
        time. This allows estimation of the covariance structure in a streaming
        way, without having to load all data into memory.
    n_outcomes : int
        The number of features modeled by this marginal model. For example,
        this corresponds to the number of genes being simulated.
    parameters : Dict[str, CovarianceStructure]
        A dictionary of CovarianceStructure objects. Each key corresponds to a
        different category specified in the original formula. The covariance
        structure stores the relationships among genes. It can be a standard
        covariance matrix, but may also use more memory-efficient approximations
        like when using CovarianceStructure with a constraint on
        num_modeled_genes.
    groups : list
        The list of groups in the formula.
    n_groups : int
        The number of groups in the formula.
    """

    def __init__(self, formula: Union[str, dict] = "~ 1"):
        """
        Initialize a :class:`StandardCopula` instance.

        Parameters
        ----------
        formula : str, optional
            Copula formula specifying categorical covariates (e.g. cell type).
            The formula is processed so that a ``"group"`` predictor is present,
            which is then used to estimate group-specific covariance matrices.
        """
        formula = standardize_formula(formula, allowed_keys=["group"])
        super().__init__(formula)
        self.groups = None
        self.copula_likelihood = 0

    def setup_data(self, adata: AnnData, marginal_formula: Dict[str, str], **kwargs):
        """
        Set up data and design matrices for copula estimation.

        After this call, the internal loader produces batches whose
        ``x_dict`` always contains a binary ``"group"`` one‑hot matrix.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with cells in rows and features (e.g. genes)
            in columns.
        marginal_formula : dict of {str: str}
            Mapping from parameter name to formula used for the marginal
            models. This is combined with the copula formula.
        **kwargs
            Additional keyword arguments passed to :func:`adata_loader`
            (e.g. ``batch_size``, shuffling, device options).

        Raises
        ------
        ValueError
            If the inferred ``"group"`` design matrix is not binary, i.e.
            contains entries other than 0 or 1.
        """
        data_kwargs = _filter_kwargs(kwargs, DEFAULT_ALLOWED_KWARGS["data"])
        super().setup_data(adata, marginal_formula, **data_kwargs)
        _, obs_batch = next(iter(self.loader))
        obs_batch_group = obs_batch.get("group")

        # fill in group indexing variables
        self.groups = self.loader.dataset.predictor_names["group"]
        self.n_groups = len(self.groups)
        self._group_col = {g: i for i, g in enumerate(self.groups)}

        # check that obs_batch is a binary grouping matrix (only if group exists)
        if obs_batch_group is not None:
            unique_vals = torch.unique(obs_batch_group)
            if not torch.all((unique_vals == 0) | (unique_vals == 1)).item():
                raise ValueError(
                    "Only categorical groups are currently supported in copula correlation estimation."
                )

    def fit(self, uniformizer: Callable, **kwargs):
        r"""
        Fit the Gaussian copula covariance model.

        The data are first transformed to pseudo‑Gaussian variables via the
        ``uniformizer`` (PIT) and an inverse normal CDF. Depending on
        ``top_k``, either a full correlation matrix is estimated for all genes,
        or a block structure with an explicit correlation matrix for the top‑``k``
        most expressed genes and independent standard normal marginals for the
        remainder.

        Parameters
        ----------
        uniformizer : callable
            Function with signature ``uniformizer(y, x_dict) -> np.ndarray``
            (or tensor convertible to ``np.ndarray``) that converts
            expression data to uniform \([0, 1]\) values.
        **kwargs
            Additional keyword arguments controlling the fit.

        Other Parameters
        ----------------
        top_k : int, optional
            Number of most expressed genes to model with a full covariance
            block. If ``None``, a full covariance matrix is estimated for all genes.

        Raises
        ------
        ValueError
            If ``top_k`` is not a positive integer or exceeds the number
            of modeled outcomes.
        """
        top_k = self._validate_parameters(**kwargs)
        modeled_indices, remaining_indices = self._get_gene_partitions(top_k)
        self.copula_likelihood = 0
        self.parameters = self._estimate_correlation(
            uniformizer, modeled_indices, remaining_indices
        )

    def pseudo_obs(self, x_dict: Dict):
        """
        Sample dependent uniform pseudo‑observations from the fitted copula.

        Parameters
        ----------
        x_dict : dict
            Dictionary of covariates for the current batch. Must contain a
            key ``"group"`` with a one‑hot matrix representing group
            memberships for each observation.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_cells, n_genes)`` containing uniform
            pseudo‑observations sampled from the fitted copula.
        """
        # convert one-hot encoding memberships to a map
        #      {"group1": [indices of group 1], "group2": [indices of group 2]}
        # The initialization method ensures that x_dict will always have a "group" key.
        group_data = x_dict.get("group")
        memberships = group_data.cpu().numpy()
        group_ix = {
            g: np.where(memberships[:, self._group_col[g]] == 1)[0] for g in self.groups
        }

        # initialize the result
        u = np.zeros((len(memberships), self.n_outcomes))
        parameters = self.parameters

        # loop over groups and sample each part in turn
        for group, cov_struct in parameters.items():
            u[group_ix[group]] = self._normal_pseudo_obs(
                len(group_ix[group]), cov_struct
            )
        return u

    def likelihood(
        self,
        uniformizer: Callable,
        batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
    ):
        """
        Compute per‑cell log‑likelihood under the fitted copula model.

        Parameters
        ----------
        uniformizer : callable
            Function that converts expression data to uniform ``[0, 1]``
            pseudo‑observations given covariates, with signature
            ``uniformizer(y, x_dict)``.
        batch : tuple of (torch.Tensor, dict)
            A mini‑batch as returned by the internal data loader, containing:

            * ``y`` (:class:`torch.Tensor`): expression data of shape
              ``(n_cells, n_genes)``.
            * ``x_dict`` (dict of str to :class:`torch.Tensor`): covariate
              matrices, including a ``"group"`` one‑hot matrix.

        Returns
        -------
        np.ndarray
            One‑dimensional array of shape (n_cells,) with the
            log‑likelihood for each observation.
        """
        # uniformize the observations
        y, x_dict = batch
        u = uniformizer(y, x_dict)
        z = norm().ppf(u)

        # same group manipulation as for pseudobs
        parameters = self.parameters
        if type(parameters) is not dict:
            parameters = {self.groups[0]: parameters}

        group_data = x_dict.get("group")
        memberships = group_data.cpu().numpy()
        group_ix = {
            g: np.where(memberships[:, self._group_col[g]] == 1)[0] for g in self.groups
        }

        ll = np.zeros(len(z))

        for group, cov_struct in parameters.items():
            ix = group_ix[group]
            if len(ix) > 0:
                z_modeled = z[ix][:, cov_struct.modeled_indices]

                self._validate_correlation_matrix(cov_struct.cov.values, group)
                ll_marginal = norm.logpdf(z_modeled).sum(axis=1)
                ll_modeled = multivariate_normal.logpdf(
                    z_modeled,
                    np.zeros(cov_struct.num_modeled_genes),
                    cov_struct.cov.values,
                    allow_singular=False,
                )
                ll[ix] = ll_modeled - ll_marginal
        return ll

    def num_params(self, **kwargs):
        """
        Return the effective number of correlation parameters.

        Parameters
        ----------
        **kwargs
            Currently unused, kept for consistency with other copula
            implementations.

        Returns
        -------
        int
            Total number of free covariance parameters across all groups,
            computed as the number of unique off‑diagonal entries in each
            modeled correlation block.
        """
        S = self.parameters
        per_group = [
            ((S[g].num_modeled_genes * (S[g].num_modeled_genes - 1)) / 2)
            for g in self.groups
        ]
        return int(sum(per_group))

    def _validate_parameters(self, **kwargs):
        """
        Internal helper to validate keyword arguments for :meth:`fit`.
        """
        top_k = kwargs.get("top_k", None)
        if top_k is not None:
            if not isinstance(top_k, int):
                raise ValueError("top_k must be an integer")
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            if top_k > self.n_outcomes:
                raise ValueError(
                    f"top_k ({top_k}) cannot exceed number of outcomes "
                    f"({self.n_outcomes})"
                )
        return top_k

    def _get_gene_partitions(
        self, top_k: Union[int, None]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return modeled and remaining gene indices for the requested fit.
        top_k selected based on order of per-gene expression level.
        """
        if top_k is None:
            return np.arange(self.n_outcomes), np.array([], dtype=int)

        gene_total_expression = np.asarray(self.adata.X.sum(axis=0)).flatten()
        sorted_indices = np.argsort(gene_total_expression)
        return sorted_indices[-top_k:], sorted_indices[:-top_k]

    def _accumulate_stats(
        self,
        uniformizer: Callable,
        modeled_indices: np.ndarray,
        remaining_indices: np.ndarray,
    ) -> Tuple[
        Dict[Union[str, int], np.ndarray],
        Dict[Union[str, int], np.ndarray],
        Dict[Union[str, int], int],
        Dict[Union[str, int], list],
    ]:
        """
        Accumulate sufficient statistics for covariance and correlation estimation.

        Parameters
        ----------
        uniformizer : callable
            Function that converts each batch of counts to uniform values.
        modeled_indices : np.ndarray
            Array of indices corresponding to selected genes.
        remaining_indices : np.ndarray
            Array of indices for the remaining genes.

        Returns
        -------
        sums : dict
            Per‑group sums of the transformed selected genes.
        second_moments : dict
            Per‑group second‑moment matrices for the selected genes.
        group_sizes : dict
            Per‑group number of observations contributing to the statistics.
        modeled_Z : dict
            Per‑group arrays of uniformized entries for the selected genes across all batches.
        """
        n_modeled = len(modeled_indices)
        sums = {g: np.zeros(n_modeled) for g in self.groups}
        second_moments = {g: np.zeros((n_modeled, n_modeled)) for g in self.groups}
        group_sizes = {g: 0 for g in self.groups}
        modeled_z = {g: [] for g in self.groups}

        desc = (
            "Estimating top-k copula correlation"
            if len(remaining_indices) > 0
            else "Estimating copula correlation"
        )
        for y, x_dict in tqdm(self.loader, desc=desc):
            memberships = x_dict["group"].cpu().numpy()
            z = norm.ppf(uniformizer(y, x_dict))

            for g in self.groups:
                mask = memberships[:, self._group_col[g]] == 1
                if not np.any(mask):
                    continue

                z_g = z[mask][:, modeled_indices]
                second_moments[g] += z_g.T @ z_g
                sums[g] += z_g.sum(axis=0)
                group_sizes[g] += z_g.shape[0]
                modeled_z[g].append(z_g)

        return sums, second_moments, group_sizes, modeled_z

    def _estimate_correlation(
        self,
        uniformizer: Callable,
        modeled_indices: np.ndarray,
        remaining_indices: np.ndarray,
    ) -> Dict[Union[str, int], CovarianceStructure]:
        """
        Compute correlation structures for genes at ``modeled_indices``, selected
        by ``top_k`` indicated in :meth:`fit`.  If ``top_k`` is 
        indicated in :meth:`fit`, correlation of the top-``k`` genes are estimated; 
        if ``top_k`` is None, correlation of the full matrix is estimated.

        Parameters
        ----------
        uniformizer : callable
            Function that converts each batch of expression counts to
            uniform values.
        modeled_indices : np.ndarray
            Array of indices corresponding to selected genes.
        remaining_indices : np.ndarray
            Array of indices for the remaining genes.

        Returns
        -------
        covariance : dict
            Mapping from group labels to :class:`CovarianceStructure`
            objects that encode the estimated covariance for each group.
        """
        sums, second_moments, group_sizes, modeled_z = self._accumulate_stats(
            uniformizer, modeled_indices, remaining_indices
        )
        covariance = {}
        remaining_var = (
            np.ones(len(remaining_indices)) if len(remaining_indices) > 0 else None
        )
        for g in self.groups:
            if group_sizes[g] == 0:
                warnings.warn(f"Group {g} has no observations, skipping")
                continue

            mean = sums[g] / group_sizes[g]
            cov = second_moments[g] / group_sizes[g] - np.outer(mean, mean)
            corr = self._covariance_to_correlation(cov)
            self._validate_correlation_matrix(corr, g)

            z_g = np.vstack(modeled_z[g])
            marginal_ll = norm.logpdf(z_g).sum()
            ll = multivariate_normal.logpdf(
                z_g,
                np.zeros(len(modeled_indices)),
                corr,
                allow_singular=False,
            ).sum()
            self.copula_likelihood += ll - marginal_ll

            covariance[g] = CovarianceStructure(
                cov=corr,
                modeled_names=self.adata.var_names[modeled_indices],
                modeled_indices=modeled_indices,
                remaining_var=remaining_var,
                remaining_indices=remaining_indices if len(remaining_indices) > 0 else None,
                remaining_names=(
                    self.adata.var_names[remaining_indices]
                    if len(remaining_indices) > 0
                    else None
                ),
            )
        return covariance

    def _covariance_to_correlation(self, cov: np.ndarray) -> np.ndarray:
        """
        Convert an empirical covariance matrix to a correlation matrix.
        """
        std = np.sqrt(np.clip(np.diag(cov), a_min=np.finfo(float).eps, a_max=None))
        corr = cov / np.outer(std, std)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _validate_correlation_matrix(self, corr: np.ndarray, group: Union[str, int]) -> None:
        """
        Validate that a correlation matrix can be used for Gaussian log-likelihood.
        """
        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The copula correlation matrix is not positive "
                f"definite for group '{group}'. Matrix shape is "
                f"{corr.shape[0]} rows x {corr.shape[1]} columns. "
                "Try setting a smaller top_k argument "
            ) from exc

    def _normal_pseudo_obs(
        self, n_samples: int, cov_struct: CovarianceStructure
    ) -> np.ndarray:
        """
        Sample uniform pseudo‑observations from a correlation structure.

        Parameters
        ----------
        n_samples : int
            Number of samples (cells) to generate.
        cov_struct : CovarianceStructure
            Covariance structure containing either a full correlation matrix
            or a modeled block plus independent standard normal remainder.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, total_genes)`` containing uniform
            pseudo‑observations.
        """
        u = np.zeros((n_samples, cov_struct.total_genes))
        z_modeled = np.random.multivariate_normal(
            mean=np.zeros(cov_struct.num_modeled_genes),
            cov=cov_struct.cov.values,
            size=n_samples,
        )
        u[:, cov_struct.modeled_indices] = norm.cdf(z_modeled)

        if cov_struct.num_remaining_genes > 0:
            z_remaining = np.random.normal(
                loc=0,
                scale=1,
                size=(n_samples, cov_struct.num_remaining_genes),
            )
            u[:, cov_struct.remaining_indices] = norm.cdf(z_remaining)

        return u
