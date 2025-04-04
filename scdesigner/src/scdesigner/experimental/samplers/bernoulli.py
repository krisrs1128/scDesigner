from scipy.stats import bernoulli
from . import glm_factory as glm
import numpy as np


def bernoulli_regression_sample_array(parameters: dict, x: np.array) -> np.array:
    theta = np.exp(x @ parameters["beta"])
    return bernoulli(theta).rvs()


def bernoulli_copula_sample_array(
    parameters: dict, x: np.array, groups: dict
) -> np.array:
    G = parameters["beta"].shape[1]
    u = np.zeros((x.shape[0], G))
    u = glm.gaussian_copula_pseudo_obs(x.shape[0], G, parameters["covariance"], groups)

    mu = np.exp(x @ parameters["beta"])
    return bernoulli(mu).ppf(u)


bernoulli_sample = glm.glm_sample_factory(
    bernoulli_regression_sample_array,
    lambda parameters: parameters["beta"].columns,
)

bernoulli_copula_sample = glm.gaussian_copula_sample_factory(
    bernoulli_copula_sample_array,
    lambda parameters: parameters["beta"].columns,
)
