from . import glm_factory as glm
from scipy.stats import bernoulli
from typing import Union
import numpy as np


def bernoulli_regression_sample_array(local_parameters: dict) -> np.array:
    theta = local_parameters["mean"]
    return bernoulli(theta).rvs()


def bernoulli_copula_sample_array(
    local_parameters: dict, covariance: Union[dict, np.array], groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
    N, G = local_parameters["mean"].shape
    u = glm.gaussian_copula_pseudo_obs(N, G, covariance, groups)

    theta = local_parameters["mean"]
    return bernoulli(theta).ppf(u)


bernoulli_sample = glm.glm_sample_factory(bernoulli_regression_sample_array)

bernoulli_copula_sample = glm.gaussian_copula_sample_factory(
    bernoulli_copula_sample_array
)
