import numpy as np
import pandas as pd
import anndata as ad
from ..estimators.glm_regression import group_indices
from scipy.stats import norm
from formulaic import model_matrix


def glm_sample_factory(sample_array):
    def sampler(parameters: dict, obs: pd.DataFrame, formula=None) -> ad.AnnData:
        if formula is not None:
            x = model_matrix(formula, obs)
        else:
            x = obs

        samples = sample_array(parameters, x)
        result = ad.AnnData(X=samples, obs=obs)
        result.var_names = parameters["dispersion"].columns
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


def gaussian_copula_sample_factory(copula_sample_array, var_names_fun):
    def sampler(
        parameters: dict, obs: pd.DataFrame, formula="~ 1", formula_copula="~ 1"
    ) -> ad.AnnData:
        x = model_matrix(formula, obs)
        groups = group_indices(formula_copula, obs)

        samples = copula_sample_array(parameters, x, groups)
        result = ad.AnnData(X=samples, obs=obs)
        result.var_names = var_names_fun(parameters)
        return result
    return sampler

