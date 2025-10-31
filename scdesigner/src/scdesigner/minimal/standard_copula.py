from .copula import Copula
from .formula import standardize_formula
from .kwargs import DEFAULT_ALLOWED_KWARGS, _filter_kwargs
from anndata import AnnData
from scipy.stats import norm, multivariate_normal
from tqdm import tqdm
from typing import Dict, Union, Callable, Tuple
import numpy as np
import torch
from .copula import CovarianceStructure
import warnings

class StandardCopula(Copula):
    """Standard Gaussian Copula Model"""
    def __init__(self, formula: str = "~ 1"):
        """Initialize the StandardCopula model.

        Args:
            formula (str, optional): _description_. Defaults to "~ 1".
        """
        formula = standardize_formula(formula, allowed_keys=['group'])
        super().__init__(formula)
        self.groups = None


    def setup_data(self, adata: AnnData, marginal_formula: Dict[str, str], **kwargs):
        """Set up the data for the standard covariance model. After setting up the data, x_dict will always have a "group" key.

        Args:
            adata (AnnData): The AnnData object containing the data.
            marginal_formula (Dict[str, str]): The formula for the marginal model.
        Raises:
            ValueError: If the groupings are not binary.
        """
        data_kwargs = _filter_kwargs(kwargs, DEFAULT_ALLOWED_KWARGS['data'])
        super().setup_data(adata, marginal_formula, **data_kwargs)
        _, obs_batch = next(iter(self.loader))
        obs_batch_group = obs_batch.get("group")

        # fill in group indexing variables
        self.groups = self.loader.dataset.predictor_names["group"]
        self.n_groups = len(self.groups)
        self.group_col = {g: i for i, g in enumerate(self.groups)}

        # check that obs_batch is a binary grouping matrix (only if group exists)
        if obs_batch_group is not None:
            unique_vals = torch.unique(obs_batch_group)
            if (not torch.all((unique_vals == 0) | (unique_vals == 1)).item()):
                raise ValueError("Only categorical groups are currently supported in copula covariance estimation.")

    def fit(self, uniformizer: Callable, **kwargs):
        """
        Fit the copula covariance model.

        Args:
            uniformizer (Callable): Function to convert data to uniform distribution
            **kwargs: Additional arguments
                top_k (int, optional): Use only top-k most expressed genes for covariance estimation.
                                    If None, estimates full covariance for all genes.

        Returns:
            None: Stores fitted parameters in self.parameters as dict of CovarianceStructure objects.

        Raises:
            ValueError: If top_k is not a positive integer or exceeds n_outcomes
        """
        top_k = kwargs.get("top_k", None)
        if top_k is not None:
            if not isinstance(top_k, int):
                raise ValueError("top_k must be an integer")
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            if top_k > self.n_outcomes:
                raise ValueError(f"top_k ({top_k}) cannot exceed number of outcomes ({self.n_outcomes})")
            gene_total_expression = np.array(self.adata.X.sum(axis=0)).flatten()
            sorted_indices = np.argsort(gene_total_expression)
            top_k_indices = sorted_indices[-top_k:]
            remaining_indices = sorted_indices[:-top_k]
            covariances = self._compute_block_covariance(uniformizer, top_k_indices,
                                                             remaining_indices, top_k)
        else:
            covariances = self._compute_full_covariance(uniformizer)

        self.parameters = covariances

    def pseudo_obs(self, x_dict: Dict):
        # convert one-hot encoding memberships to a map
        #      {"group1": [indices of group 1], "group2": [indices of group 2]}
        # The initialization method ensures that x_dict will always have a "group" key.
        group_data = x_dict.get("group")
        memberships = group_data.cpu().numpy()
        group_ix = {g: np.where(memberships[:, self.group_col[g] == 1])[0] for g in self.groups}

        # initialize the result
        u = np.zeros((len(memberships), self.n_outcomes))
        parameters = self.parameters

        # loop over groups and sample each part in turn
        for group, cov_struct in parameters.items():
            if cov_struct.remaining_var is not None:
                u[group_ix[group]] = self._fast_normal_pseudo_obs(len(group_ix[group]), cov_struct)
            else:
                u[group_ix[group]] = self._normal_pseudo_obs(len(group_ix[group]), cov_struct)
        return u

    def likelihood(self, uniformizer: Callable, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]):
        """
        Compute likelihood of data given the copula model.

        Args:
            uniformizer (Callable): Function to convert expression data to uniform distribution
            batch (Tuple[torch.Tensor, Dict[str, torch.Tensor]]): Data batch containing:
                - Y (torch.Tensor): Expression data of shape (n_cells, n_genes)
                - X_dict (Dict[str, torch.Tensor]): Covariates dict with keys as parameter names
                                                and values as tensors of shape (n_cells, n_covariates)

        Returns:
            np.ndarray: Log-likelihood for each cell, shape (n_cells,)
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
        memberships = group_data.numpy()
        group_ix = {g: np.where(memberships[:, self.group_col[g] == 1])[0] for g in self.groups}

        ll = np.zeros(len(z))

        for group, cov_struct in parameters.items():
            ix = group_ix[group]
            if len(ix) > 0:
                z_modeled = z[ix][:, cov_struct.modeled_indices]

                ll_modeled = multivariate_normal.logpdf(z_modeled,
                                                       np.zeros(cov_struct.num_modeled_genes),
                                                       cov_struct.cov.values)
                if cov_struct.num_remaining_genes > 0:
                    z_remaining = z[ix][:, cov_struct.remaining_indices]
                    ll_remaining = norm.logpdf(z_remaining,
                                            loc=0,
                                            scale = np.sqrt(cov_struct.remaining_var.values))
                else:
                    ll_remaining = 0
                ll[ix] = ll_modeled + ll_remaining
        return ll

    def num_params(self, **kwargs):
        S = self.parameters
        per_group = [((S[g].num_modeled_genes * (S[g].num_modeled_genes - 1)) / 2) for g in self.groups]
        return sum(per_group)

    def _validate_parameters(self, **kwargs):
        top_k = kwargs.get("top_k", None)
        if top_k is not None:
            if not isinstance(top_k, int):
                raise ValueError("top_k must be an integer")
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            if top_k > self.n_outcomes:
                raise ValueError(f"top_k ({top_k}) cannot exceed number of outcomes ({self.n_outcomes})")
        return top_k



    def _accumulate_top_k_stats(self, uniformizer:Callable, top_k_idx, rem_idx, top_k) \
        -> Tuple[Dict[Union[str, int], np.ndarray],
                 Dict[Union[str, int], np.ndarray],
                 Dict[Union[str, int], np.ndarray],
                 Dict[Union[str, int], np.ndarray],
                 Dict[Union[str, int], int]]:
        """Accumulate sufficient statistics for top-k covariance estimation.

        Args:
            uniformizer (Callable): Function to convert to uniform distribution
            top_k_idx (np.ndarray): Indices of the top-k genes
            rem_idx (np.ndarray): Indices of the remaining genes
            top_k (int): Number of top-k genes

        Returns:
            top_k_sums (dict): Sums of the top-k genes for each group
            top_k_second_moments (dict): Second moments of the top-k genes for each group
            rem_sums (dict): Sums of the remaining genes for each group
            rem_second_moments (dict): Second moments of the remaining genes for each group
            Ng (dict): Number of observations for each group
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
                mask = memberships[:, self.group_col[g]] == 1
                if not np.any(mask):
                    continue

                z_g = z[mask]
                n_g = mask.sum()

                top_k_z, rem_z = z_g[:, top_k_idx], z_g[:, rem_idx]

                top_k_sums[g] += top_k_z.sum(axis=0)
                top_k_second_moments[g] += top_k_z.T @ top_k_z

                rem_sums[g] += rem_z.sum(axis=0)
                rem_second_moments[g] += (rem_z ** 2).sum(axis=0)

                Ng[g] += n_g

        return top_k_sums, top_k_second_moments, rem_sums, rem_second_moments, Ng

    def _accumulate_full_stats(self, uniformizer:Callable) \
        -> Tuple[Dict[Union[str, int], np.ndarray],
                 Dict[Union[str, int], np.ndarray],
                 Dict[Union[str, int], int]]:
        """Accumulate sufficient statistics for full covariance estimation.

        Args:
            uniformizer (Callable): Function to convert to uniform distribution

        Returns:
            sums (dict): Sums of the genes for each group
            second_moments (dict): Second moments of the genes for each group
            Ng (dict): Number of observations for each group
        """
        sums = {g: np.zeros(self.n_outcomes) for g in self.groups}
        second_moments = {g: np.zeros((self.n_outcomes, self.n_outcomes)) for g in self.groups}
        Ng = {g: 0 for g in self.groups}

        for y, x_dict in tqdm(self.loader, desc="Estimating copula covariance"):
            group_data = x_dict.get("group")
            memberships = group_data.cpu().numpy()

            u = uniformizer(y, x_dict)
            z = norm.ppf(u)

            for g in self.groups:
                mask = memberships[:, self.group_col[g]] == 1

                if not np.any(mask):
                    continue

                z_g = z[mask]
                n_g = mask.sum()

                second_moments[g] += z_g.T @ z_g
                sums[g] += z_g.sum(axis=0)

                Ng[g] += n_g

        return sums, second_moments, Ng

    def _compute_block_covariance(self, uniformizer:Callable,
                                  top_k_idx: np.ndarray, rem_idx: np.ndarray, top_k: int) \
        -> Dict[Union[str, int], CovarianceStructure]:
        """Compute the covariance matrix for the top-k and remaining genes.

        Args:
            top_k_sums (dict): Sums of the top-k genes for each group
            top_k_second_moments (dict): Second moments of the top-k genes for each group
            remaining_sums (dict): Sums of the remaining genes for each group
            remaining_second_moments (dict): Second moments of the remaining genes for each group
            Ng (dict): Number of observations for each group

        Returns:
            covariance (dict): Covariance matrix for each group
        """
        top_k_sums, top_k_second_moments, remaining_sums, remaining_second_moments, Ng \
            = self._accumulate_top_k_stats(uniformizer, top_k_idx, rem_idx, top_k)
        covariance = {}
        for g in self.groups:
            if Ng[g] == 0:
                warnings.warn(f"Group {g} has no observations, skipping")
                continue
            mean_top_k = top_k_sums[g] / Ng[g]
            cov_top_k = top_k_second_moments[g] / Ng[g] - np.outer(mean_top_k, mean_top_k)
            mean_remaining = remaining_sums[g] / Ng[g]
            var_remaining = remaining_second_moments[g] / Ng[g] - mean_remaining ** 2
            top_k_names = self.adata.var_names[top_k_idx]
            remaining_names = self.adata.var_names[rem_idx]
            covariance[g] = CovarianceStructure(
                cov=cov_top_k,
                modeled_names=top_k_names,
                modeled_indices=top_k_idx,
                remaining_var=var_remaining,
                remaining_indices=rem_idx,
                remaining_names=remaining_names
            )
        return covariance

    def _compute_full_covariance(self, uniformizer:Callable) -> Dict[Union[str, int], CovarianceStructure]:
        """Compute the covariance matrix for the full genes.

        Args:
            uniformizer (Callable): Function to convert to uniform distribution

        Returns:
            covariance (dict): Covariance matrix for each group
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
                remaining_names=None
            )
        return covariance

    def _fast_normal_pseudo_obs(self, n_samples: int, cov_struct: CovarianceStructure) -> np.ndarray:
        """Sample pseudo-observations from the covariance structure.

        Args:
            n_samples (int): Number of samples to generate
            cov_struct (CovarianceStructure): The covariance structure

        Returns:
            np.ndarray: Pseudo-observations with shape (n_samples, total_genes)
        """
        u = np.zeros((n_samples, cov_struct.total_genes))

        z_modeled = np.random.multivariate_normal(
            mean=np.zeros(cov_struct.num_modeled_genes),
            cov=cov_struct.cov.values,
            size=n_samples
        )

        z_remaining = np.random.normal(
            loc=0,
            scale=cov_struct.remaining_var.values ** 0.5,
            size=(n_samples, cov_struct.num_remaining_genes)
        )

        normal_distn_modeled = norm(0, np.diag(cov_struct.cov.values) ** 0.5)
        u[:, cov_struct.modeled_indices] = normal_distn_modeled.cdf(z_modeled)

        normal_distn_remaining = norm(0, cov_struct.remaining_var.values ** 0.5)
        u[:, cov_struct.remaining_indices] = normal_distn_remaining.cdf(z_remaining)

        return u

    def _normal_pseudo_obs(self, n_samples: int, cov_struct: CovarianceStructure) -> np.ndarray:
        """Sample pseudo-observations from the covariance structure.

        Args:
            n_samples (int): Number of samples to generate
            cov_struct (CovarianceStructure): The covariance structure

        Returns:
            np.ndarray: Pseudo-observations with shape (n_samples, total_genes)
        """
        u = np.zeros((n_samples, cov_struct.total_genes))
        z = np.random.multivariate_normal(
            mean=np.zeros(cov_struct.total_genes),
            cov=cov_struct.cov.values,
            size=n_samples
        )

        normal_distn = norm(0, np.diag(cov_struct.cov.values) ** 0.5)
        u = normal_distn.cdf(z)

        return u