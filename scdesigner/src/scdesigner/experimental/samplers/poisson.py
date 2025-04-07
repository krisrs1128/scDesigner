from . import glm_factory as glm
from scipy.stats import poisson
from typing import Union
import numpy as np


def poisson_regression_sample_array(local_parameters: dict) -> np.array:
    mu = local_parameters["mean"]
    return poisson(mu).rvs()


def poisson_copula_sample_array(
    local_parameters: dict, covariance: Union[dict, np.array], groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
    N, G = local_parameters["mean"].shape
    u = glm.gaussian_copula_pseudo_obs(N, G, covariance, groups)

    # invert using poisson margins
    mu = local_parameters["mean"]
    return poisson(mu).ppf(u)


poisson_sample = glm.glm_sample_factory(poisson_regression_sample_array)
poisson_copula_sample = glm.gaussian_copula_sample_factory(poisson_copula_sample_array)
