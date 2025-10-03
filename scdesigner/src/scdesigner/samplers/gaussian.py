from scipy.stats import norm
from . import glm_factory as glm
from typing import Union
import numpy as np


def gaussian_regression_sample_array(local_parameters: dict) -> np.array:
    sigma, mu = local_parameters["sdev"], local_parameters["mean"] # dataframes of shape (n, g)
    return norm(loc=mu, scale=sigma).rvs()


def gaussian_copula_sample_array(
    local_parameters: dict, covariance: Union[dict, np.array], groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
    N, G = local_parameters["mean"].shape
    u = glm.gaussian_copula_pseudo_obs(N, G, covariance, groups)

    # transform the correlated uniforms to NB space
    sigma, mu = local_parameters["sdev"], local_parameters["mean"]
    return norm(loc=mu, scale=sigma).ppf(u)


gaussian_regression_sample = glm.glm_sample_factory(gaussian_regression_sample_array)
gaussian_copula_sample = glm.gaussian_copula_sample_factory(gaussian_copula_sample_array)
