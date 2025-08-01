import numpy as np
import pandas as pd
import anndata as ad
from typing import Union
from scipy.stats import norm


def glm_sample_factory(sample_array):
    def sampler(local_parameters: dict, obs: pd.DataFrame) -> ad.AnnData:
        samples = sample_array(local_parameters)
        result = ad.AnnData(X=samples, obs=obs)
        result.var_names = local_parameters["mean"].columns
        return result
    return sampler

def gaussian_copula_pseudo_obs(N, G, sigma, groups):
    u = np.zeros((N, G))

    # cycle across groups
    for group, ix in groups.items():
        if type(sigma) is not dict:
            sigma = {group: sigma}

        z = np.random.multivariate_normal(
            mean=np.zeros(G), cov=sigma[group], size=len(ix)
        )
        normal_distn = norm(0, np.diag(sigma[group] ** 0.5))
        u[ix] = normal_distn.cdf(z)
    return u


def gaussian_copula_sample_factory(copula_sample_array):
    def sampler(
        local_parameters: dict, covariance: Union[dict, np.array], groups: dict, obs: pd.DataFrame
    ) -> ad.AnnData:
        samples = copula_sample_array(local_parameters, covariance, groups)
        result = ad.AnnData(X=samples, obs=obs)
        result.var_names = local_parameters["mean"].columns
        return result
    return sampler

