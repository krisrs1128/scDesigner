from scipy.stats import nbinom
from . import glm_factory as glm
from typing import Union
import numpy as np


def negbin_regression_sample_array(local_parameters: dict) -> np.array:
    r, mu = local_parameters["dispersion"], local_parameters["mean"] # dataframes of shape (n, g)
    return nbinom(n=r, p=r / (r + mu)).rvs()


def negbin_copula_sample_array(
    local_parameters: dict, covariance: Union[dict, np.array], groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
    N, G = local_parameters["mean"].shape
    u = glm.gaussian_copula_pseudo_obs(N, G, covariance, groups)

    # transform the correlated uniforms to NB space
    r, mu = local_parameters["dispersion"], local_parameters["mean"]
    return nbinom(n=r, p=r / (r + mu)).ppf(u)


negbin_sample = glm.glm_sample_factory(negbin_regression_sample_array)
negbin_copula_sample = glm.gaussian_copula_sample_factory(negbin_copula_sample_array)
