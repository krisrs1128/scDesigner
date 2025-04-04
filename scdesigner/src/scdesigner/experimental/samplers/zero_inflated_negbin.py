from scipy.stats import nbinom, bernoulli
from . import glm_factory as glm
import numpy as np


def zero_inflated_negbin_sample_array(parameters: dict, x: np.array) -> np.array:
    r, mu, pi = np.exp(parameters["dispersion"]), np.exp(x @ parameters["beta"]), parameters["pi"]
    r = np.repeat(r, mu.shape[0], axis=0)
    return nbinom(n=r, p=r / (r + mu)).rvs() * bernoulli(pi).rvs()


def zero_inflated_negbin_copula_sample_array(
    parameters: dict, x: np.array, groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
    G = parameters["beta"].shape[1]
    u = np.zeros((x.shape[0], G))
    u = glm.gaussian_copula_pseudo_obs(x.shape[0], G, parameters["covariance"], groups)

    # get zero inflated NB parameters
    r, mu, pi = np.exp(parameters["dispersion"]), np.exp(x @ parameters["beta"]), parameters["pi"]
    r = np.repeat(r, mu.shape[0], axis=0)

    # zero inflate after first simulating from NB
    positive_part = nbinom(n=r, p=r / (r + mu)).ppf(u)
    zero_inflation = bernoulli(pi).ppf(u)
    return zero_inflation * positive_part


zero_inflated_negbin_sample = glm.glm_sample_factory(
    zero_inflated_negbin_sample_array,
    lambda parameters: parameters["dispersion"].columns,
)

zero_inflated_negbin_copula_sample = glm.gaussian_copula_sample_factory(
    zero_inflated_negbin_copula_sample_array,
    lambda parameters: parameters["dispersion"].columns,
)
