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
        sums = {g: np.zeros(self.n_outcomes) for g in self.groups}
        second_moments = {g: np.eye(self.n_outcomes) for g in self.groups}
        Ng = {g: 0 for g in self.groups}

        for y, x_dict in tqdm(self.loader, desc="Estimating copula covariance"):
            memberships = x_dict.get("group").numpy()
            u = uniformizer(y, x_dict)

            for g in self.groups:
                ix = np.where(memberships[:, self.group_col[g]] == 1)
                z = norm().ppf(u[ix])
                second_moments[g] += z.T @ z
                sums[g] += z.sum(axis=0)
                Ng[g] += len(ix[0])

        covariances = {}
        for g in self.groups:
            mean = sums[g] / Ng[g]
            covariances[g] = second_moments[g] / Ng[g] - np.outer(mean, mean)

        if len(self.groups) == 1:
            covariances = list(covariances.values())[0]
        self.parameters = self.format_parameters(covariances)

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