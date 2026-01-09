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
    Gaussian copula model with optional group-specific covariance structures.

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

    Examples
    --------
    >>> import numpy as np
    >>> import scanpy as sc
    >>> from scdesigner.copulas.standard_copula import StandardCopula
    >>>
    >>> # Load a small dataset (cells x genes) and keep only a few genes for speed
    >>> adata = sc.datasets.pbmc3k()[:500, :20].copy()
    >>>
    >>> # Instantiate the copula with a simple group formula and set up data
    >>> copula = StandardCopula("group ~ 1")
    >>> copula.setup_data(adata, {"group": "~ 1"}, batch_size=256)
    >>> copula.groups  # groups inferred from the design matrix
    ['Intercept']
    >>> copula.n_outcomes  # number of modeled genes
    20
    >>> # Define a simple rank-based uniformizer used by fit() and likelihood()
    >>> def rank_uniformizer(y, x_dict):
    ...     y_np = y.cpu().numpy()
    ...     # Convert each gene to ranks and scale to (0, 1)
    ...     ranks = np.argsort(np.argsort(y_np, axis=0), axis=0) + 1
    ...     return ranks / (y_np.shape[0] + 1.0)
    >>>
    >>> # Fit the Gaussian copula covariance model
    >>> copula.fit(rank_uniformizer, top_k=10)
    >>> isinstance(copula.parameters, dict)
    True
    >>> # Draw dependent uniform pseudo-observations for a batch of covariates
    >>> y_batch, x_batch = next(iter(copula.loader))
    >>> u = copula.pseudo_obs(x_batch)
    >>> u.shape[1] == copula.n_outcomes
    True
    >>> # Compute per-cell log-likelihoods for the same batch
    >>> ll = copula.likelihood(rank_uniformizer, (y_batch, x_batch))
    >>> ll.shape[0] == y_batch.shape[0]
    True
    >>> # Inspect the effective number of covariance parameters
    >>> n_params = copula.num_params()
    >>> isinstance(n_params, int) and n_params > 0
    True

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

    def setup_data(self, adata: AnnData, marginal_formula: Dict[str, str], **kwargs):
        """
        Set up data and design matrices for covariance estimation.

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
                    "Only categorical groups are currently supported in copula covariance estimation."
                )

    def fit(self, uniformizer: Callable, **kwargs):
        """
        Fit the Gaussian copula covariance model.

        The data are first transformed to pseudo‑Gaussian variables via the
        ``uniformizer`` (PIT) and an inverse normal CDF. Depending on
        ``top_k``, either a full covariance matrix is estimated for all genes,
        or a block structure with an explicit covariance for the top‑``k``
        most expressed genes and diagonal variances for the remainder.

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
            gene_total_expression = np.array(self.adata.X.sum(axis=0)).flatten()
            sorted_indices = np.argsort(gene_total_expression)
            top_k_indices = sorted_indices[-top_k:]
            remaining_indices = sorted_indices[:-top_k]
            covariances = self._compute_block_covariance(
                uniformizer, top_k_indices, remaining_indices, top_k
            )
        else:
            covariances = self._compute_full_covariance(uniformizer)

        self.parameters = covariances

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
            if cov_struct.remaining_var is not None:
                u[group_ix[group]] = self._fast_normal_pseudo_obs(
                    len(group_ix[group]), cov_struct
                )
            else:
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
            Function that converts expression data to uniform \([0, 1]\)
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

                ll_modeled = multivariate_normal.logpdf(
                    z_modeled,
                    np.zeros(cov_struct.num_modeled_genes),
                    cov_struct.cov.values,
                )
                if cov_struct.num_remaining_genes > 0:
                    z_remaining = z[ix][:, cov_struct.remaining_indices]
                    ll_remaining = norm.logpdf(
                        z_remaining,
                        loc=0,
                        scale=np.sqrt(cov_struct.remaining_var.values),
                    )
                else:
                    ll_remaining = 0
                ll[ix] = ll_modeled + ll_remaining
        return ll

    def num_params(self, **kwargs):
        """
        Return the effective number of covariance parameters.

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
            modeled covariance block.
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

    def _accumulate_top_k_stats(
        self, uniformizer: Callable, top_k_idx, rem_idx, top_k
    ) -> Tuple[
        Dict[Union[str, int], np.ndarray],
        Dict[Union[str, int], np.ndarray],
        Dict[Union[str, int], np.ndarray],
        Dict[Union[str, int], np.ndarray],
        Dict[Union[str, int], int],
    ]:
        """
        Accumulate sufficient statistics for top‑``k`` block covariance.

        Parameters
        ----------
        uniformizer : callable
            Function that converts each batch of counts to uniform values.
        top_k_idx : np.ndarray
            Array of indices corresponding to the top‑``k`` genes.
        rem_idx : np.ndarray
            Array of indices for the remaining genes.
        top_k : int
            Number of top genes modeled with a full covariance block.

        Returns
        -------
        top_k_sums : dict
            Per‑group sums of the transformed top‑``k`` genes.
        top_k_second_moments : dict
            Per‑group second‑moment matrices for the top‑``k`` genes.
        rem_sums : dict
            Per‑group sums for the remaining genes.
        rem_second_moments : dict
            Per‑group sums of squared values for the remaining genes.
        Ng : dict
            Per‑group number of observations contributing to the statistics.
        """
        top_k_sums = {g: np.zeros(top_k) for g in self.groups}
        top_k_second_moments = {g: np.zeros((top_k, top_k)) for g in self.groups}
        rem_sums = {g: np.zeros(self.n_outcomes - top_k) for g in self.groups}
        rem_second_moments = {g: np.zeros(self.n_outcomes - top_k) for g in self.groups}
        Ng = {g: 0 for g in self.groups}

        for y, x_dict in tqdm(self.loader, desc="Estimating top-k copula covariance"):
            group_data = x_dict.get("group")
            memberships = group_data.cpu().numpy()
            u = uniformizer(y, x_dict)
            z = norm.ppf(u)

            for g in self.groups:
                mask = memberships[:, self._group_col[g]] == 1
                if not np.any(mask):
                    continue

                z_g = z[mask]
                n_g = mask.sum()

                top_k_z, rem_z = z_g[:, top_k_idx], z_g[:, rem_idx]

                top_k_sums[g] += top_k_z.sum(axis=0)
                top_k_second_moments[g] += top_k_z.T @ top_k_z

                rem_sums[g] += rem_z.sum(axis=0)
                rem_second_moments[g] += (rem_z**2).sum(axis=0)

                Ng[g] += n_g

        return top_k_sums, top_k_second_moments, rem_sums, rem_second_moments, Ng

    def _accumulate_full_stats(
        self, uniformizer: Callable
    ) -> Tuple[
        Dict[Union[str, int], np.ndarray],
        Dict[Union[str, int], np.ndarray],
        Dict[Union[str, int], int],
    ]:
        """
        Accumulate sufficient statistics for full covariance estimation.

        Parameters
        ----------
        uniformizer : callable
            Function that converts each batch of expression counts to
            uniform values.

        Returns
        -------
        sums : dict
            Per‑group sums of transformed values for all genes.
        second_moments : dict
            Per‑group second‑moment matrices for all genes.
        Ng : dict
            Per‑group number of observations contributing to the statistics.
        """
        sums = {g: np.zeros(self.n_outcomes) for g in self.groups}
        second_moments = {
            g: np.zeros((self.n_outcomes, self.n_outcomes)) for g in self.groups
        }
        Ng = {g: 0 for g in self.groups}

        for y, x_dict in tqdm(self.loader, desc="Estimating copula covariance"):
            group_data = x_dict.get("group")
            memberships = group_data.cpu().numpy()

            u = uniformizer(y, x_dict)
            z = norm.ppf(u)

            for g in self.groups:
                mask = memberships[:, self._group_col[g]] == 1

                if not np.any(mask):
                    continue

                z_g = z[mask]
                n_g = mask.sum()

                second_moments[g] += z_g.T @ z_g
                sums[g] += z_g.sum(axis=0)

                Ng[g] += n_g

        return sums, second_moments, Ng

    def _compute_block_covariance(
        self,
        uniformizer: Callable,
        top_k_idx: np.ndarray,
        rem_idx: np.ndarray,
        top_k: int,
    ) -> Dict[Union[str, int], CovarianceStructure]:
        """
        Compute block covariance structures for top‑``k`` and remaining genes.

        Parameters
        ----------
        uniformizer : callable
            Function that converts each batch of expression counts to
            uniform values.
        top_k_idx : np.ndarray
            Indices of the top‑``k`` genes in the original feature ordering.
        rem_idx : np.ndarray
            Indices of the remaining genes in the original feature ordering.
        top_k : int
            Number of top genes modeled with a full covariance block.

        Returns
        -------
        dict
            Mapping from group labels to :class:`CovarianceStructure`
            objects that encode the estimated covariance for each group.
        """
        (
            top_k_sums,
            top_k_second_moments,
            remaining_sums,
            remaining_second_moments,
            Ng,
        ) = self._accumulate_top_k_stats(uniformizer, top_k_idx, rem_idx, top_k)
        covariance = {}
        for g in self.groups:
            if Ng[g] == 0:
                warnings.warn(f"Group {g} has no observations, skipping")
                continue
            mean_top_k = top_k_sums[g] / Ng[g]
            cov_top_k = top_k_second_moments[g] / Ng[g] - np.outer(
                mean_top_k, mean_top_k
            )
            mean_remaining = remaining_sums[g] / Ng[g]
            var_remaining = remaining_second_moments[g] / Ng[g] - mean_remaining**2
            top_k_names = self.adata.var_names[top_k_idx]
            remaining_names = self.adata.var_names[rem_idx]
            covariance[g] = CovarianceStructure(
                cov=cov_top_k,
                modeled_names=top_k_names,
                modeled_indices=top_k_idx,
                remaining_var=var_remaining,
                remaining_indices=rem_idx,
                remaining_names=remaining_names,
            )
        return covariance

    def _compute_full_covariance(
        self, uniformizer: Callable
    ) -> Dict[Union[str, int], CovarianceStructure]:
        """
        Compute full covariance matrices for all genes.

        Parameters
        ----------
        uniformizer : callable
            Function that converts each batch of expression counts to
            uniform values.

        Returns
        -------
        dict
            Mapping from group labels to :class:`CovarianceStructure`
            objects, each containing a full covariance matrix for all genes.
        """
        sums, second_moments, Ng = self._accumulate_full_stats(uniformizer)
        covariance = {}
        for g in self.groups:
            if Ng[g] == 0:
                warnings.warn(f"Group {g} has no observations, skipping")
                continue
            mean = sums[g] / Ng[g]
            cov = second_moments[g] / Ng[g] - np.outer(mean, mean)
            covariance[g] = CovarianceStructure(
                cov=cov,
                modeled_names=self.adata.var_names,
                modeled_indices=np.arange(self.n_outcomes),
                remaining_var=None,
                remaining_indices=None,
                remaining_names=None,
            )
        return covariance

    def _fast_normal_pseudo_obs(
        self, n_samples: int, cov_struct: CovarianceStructure
    ) -> np.ndarray:
        """
        Sample uniform pseudo‑observations using a block covariance structure.

        Parameters
        ----------
        n_samples : int
            Number of samples (cells) to generate.
        cov_struct : CovarianceStructure
            Covariance structure with a modeled block and diagonal variances
            for remaining genes.

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

        z_remaining = np.random.normal(
            loc=0,
            scale=cov_struct.remaining_var.values**0.5,
            size=(n_samples, cov_struct.num_remaining_genes),
        )

        normal_distn_modeled = norm(0, np.diag(cov_struct.cov.values) ** 0.5)
        u[:, cov_struct.modeled_indices] = normal_distn_modeled.cdf(z_modeled)

        normal_distn_remaining = norm(0, cov_struct.remaining_var.values**0.5)
        u[:, cov_struct.remaining_indices] = normal_distn_remaining.cdf(z_remaining)

        return u

    def _normal_pseudo_obs(
        self, n_samples: int, cov_struct: CovarianceStructure
    ) -> np.ndarray:
        """
        Sample uniform pseudo‑observations from a full covariance matrix.

        Parameters
        ----------
        n_samples : int
            Number of samples (cells) to generate.
        cov_struct : CovarianceStructure
            Covariance structure containing a full covariance matrix
            for all genes.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, total_genes)`` containing uniform
            pseudo‑observations.
        """
        u = np.zeros((n_samples, cov_struct.total_genes))
        z = np.random.multivariate_normal(
            mean=np.zeros(cov_struct.total_genes),
            cov=cov_struct.cov.values,
            size=n_samples,
        )

        normal_distn = norm(0, np.diag(cov_struct.cov.values) ** 0.5)
        u = normal_distn.cdf(z)

        return u
