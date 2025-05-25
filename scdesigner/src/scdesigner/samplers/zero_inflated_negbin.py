from scipy.stats import nbinom, bernoulli
from . import glm_factory as glm
from typing import Union
import numpy as np


def zero_inflated_negbin_sample_array(local_parameters: dict) -> np.array:
    mu, r, pi = (
        local_parameters["mean"],
        local_parameters["dispersion"],
        local_parameters["zero_inflation"],
    )
    return nbinom(n=r, p=r / (r + mu)).rvs() * bernoulli(1 - pi).rvs()


def zero_inflated_negbin_copula_sample_array(
    local_parameters: dict, covariance: Union[dict, np.array], groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
    N, G = local_parameters["mean"].shape
    u = glm.gaussian_copula_pseudo_obs(N, G, covariance, groups)

    # get zero inflated NB parameters
    mu, r, pi = (
        local_parameters["mean"],
        local_parameters["dispersion"],
        local_parameters["zero_inflation"],
    )

    # zero inflate after first simulating from NB
    positive_part = nbinom(n=r, p=r / (r + mu)).ppf(u)
    zero_inflation = bernoulli(1 - pi).ppf(u)
    return zero_inflation * positive_part


zero_inflated_negbin_sample = glm.glm_sample_factory(zero_inflated_negbin_sample_array)

zero_inflated_negbin_copula_sample = glm.gaussian_copula_sample_factory(
    zero_inflated_negbin_copula_sample_array
)
