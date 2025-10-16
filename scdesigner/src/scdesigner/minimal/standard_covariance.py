from .copula import Copula
from .formula import standardize_formula
from .kwargs import DEFAULT_ALLOWED_KWARGS, _filter_kwargs
from anndata import AnnData
from scipy.stats import norm, multivariate_normal
from tqdm import tqdm
from typing import Dict, Union, Callable, Tuple
import numpy as np
import pandas as pd
import torch
from .copula import FastCovarianceStructure
from typing import Optional
import warnings


class StandardCovariance(Copula):
    def __init__(self, formula: str = "~ 1"):
        formula = standardize_formula(formula, allowed_keys=['group'])
        super().__init__(formula)
        self.groups = None


    def setup_data(self, adata: AnnData, marginal_formula: Dict[str, str], **kwargs):
        data_kwargs = _filter_kwargs(kwargs, DEFAULT_ALLOWED_KWARGS['data'])
        super().setup_data(adata, marginal_formula, **data_kwargs)
        _, obs_batch = next(iter(self.loader))
        obs_batch_group = obs_batch.get("group")

        # fill in group indexing variables
        self.groups = self.loader.dataset.predictor_names["group"]
        self.n_groups = len(self.groups)
        self.group_col = {g: i for i, g in enumerate(self.groups)}

        # check that obs_batch is a binary grouping matrix
        unique_vals = torch.unique(obs_batch_group)
        if (not torch.all((unique_vals == 0) | (unique_vals == 1)).item()):
            raise ValueError("Only categorical groups are currently supported in copula covariance estimation.")

    def fit(self, uniformizer: Callable, **kwargs):
        """
        Fit the copula covariance model. 
        If top_k is provided, compute the covariance matrix for the top-k most prevalent genes and self.parameters will be a dictionary of FastCovarianceStructure objects.
        Otherwise, compute the covariance matrix for the full genes and self.parameters will be a dictionary of np.ndarray objects.

        Args:
            uniformizer (Callable): Function to convert to uniform distribution
            **kwargs: Additional keyword arguments, may include top_k

        Raises:
            ValueError: If top_k is not an integer
            ValueError: If top_k is not positive
            ValueError: If top_k exceeds the number of outcomes

        """
        top_k = kwargs.get("top_k", None)
        if top_k is not None:
            if not isinstance(top_k, int):
                raise ValueError("top_k must be an integer")
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            if top_k > self.n_outcomes:
                raise ValueError(f"top_k ({top_k}) cannot exceed number of outcomes ({self.n_outcomes})")
            gene_total_expression = self.adata.X.sum(axis=0)
            top_k_indices = np.argsort(gene_total_expression)[-top_k:]
            remaining_indices = np.argsort(gene_total_expression)[:-top_k]
            self.parameters = self._compute_block_covariance(uniformizer, top_k_indices, 
                                                             remaining_indices, top_k)
        else:
            self.parameters = self._compute_full_covariance(uniformizer)
            
    def format_parameters(self, covariances: Union[Dict, np.array]):
        var_names = self.adata.var_names
        def to_df(mat):
            return pd.DataFrame(mat, index=var_names, columns=var_names)

        if isinstance(covariances, dict):
            formatted = {}
            for k, v in covariances.items():
                formatted[k] = to_df(v)
            covariances = formatted
            return covariances

        if isinstance(covariances, (np.ndarray, list, tuple)):
            covariances = to_df(covariances)
        return covariances

    def pseudo_obs(self, x_dict: Dict):
        # convert one-hot encoding memberships to a map
        #      {"group1": [indices of group 1], "group2": [indices of group 2]}
        memberships = x_dict.get("group").numpy()
        group_ix = {g: np.where(memberships[:, self.group_col[g] == 1])[0] for g in self.groups}

        # initialize the result
        u = np.zeros((len(memberships), self.n_outcomes))
        parameters = self.parameters
        if type(parameters) is not dict:
            parameters = {group: parameters}

        # loop over groups and sample each part in turn
        for group, sigma in parameters.items():
            z = np.random.multivariate_normal(
                mean=np.zeros(self.n_outcomes),
                cov=sigma,
                size=len(group_ix[group])
            )
            normal_distn = norm(0, np.diag(sigma) ** 0.5)
            u[group_ix[group]] = normal_distn.cdf(z)
        return u

    def likelihood(self, uniformizer: Callable, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]):
        # uniformize the observations
        y, x_dict = batch
        u = uniformizer(y, x_dict)
        z = norm().ppf(u)

        # same group manipulation as for pseudobs
        parameters = self.parameters
        if type(parameters) is not dict:
            parameters = {group: parameters}

        memberships = x_dict.get("group").numpy()
        group_ix = {g: np.where(memberships[:, self.group_col[g] == 1])[0] for g in self.groups}
        ll = np.zeros(len(z))
        for group, sigma in parameters.items():
            ix = group_ix[group]
            if len(ix) > 0:
                copula_ll = multivariate_normal.logpdf(z[ix], np.zeros(sigma.shape[0]), sigma)
                ll[ix] = copula_ll - norm.logpdf(z[ix]).sum(axis=1)
        return ll

    def num_params(self, **kwargs):
        S = self.parameters
        per_group = [(np.sum(S[g].values != 0) - S[g].shape[0]) / 2 for g in self.groups]
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
        top_k_second_moments = {g: np.eye(top_k) for g in self.groups}
        rem_sums = {g: np.zeros(self.n_outcomes - top_k) for g in self.groups}
        rem_second_moments = {g: np.zeros(self.n_outcomes - top_k) for g in self.groups}
        Ng = {g: 0 for g in self.groups}

        for y, x_dict in tqdm(self.loader, desc="Estimating top-k copula covariance"):
            memberships = x_dict.get("group").numpy()
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
        second_moments = {g: np.eye(self.n_outcomes) for g in self.groups}
        Ng = {g: 0 for g in self.groups}

        for y, x_dict in tqdm(self.loader, desc="Estimating copula covariance"):
            memberships = x_dict.get("group").numpy()
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
        -> Dict[Union[str, int], FastCovarianceStructure]:
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
            covariance[g] = FastCovarianceStructure(
                top_k_cov=cov_top_k,
                remaining_var=var_remaining,
                top_k_indices=top_k_idx,
                remaining_indices=rem_idx,
            )
        return covariance

    def _compute_full_covariance(self, uniformizer:Callable) -> Dict[Union[str, int], np.ndarray]:
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
            covariance[g] = second_moments[g] / Ng[g] - np.outer(mean, mean)
        return covariance
            