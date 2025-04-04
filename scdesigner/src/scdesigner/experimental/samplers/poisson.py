from scipy.stats import poisson
from . import glm_factory as glm
import numpy as np


def poisson_regression_sample_array(parameters: dict, x: np.array) -> np.array:
    mu = np.exp(x @ parameters["beta"])
    return poisson(mu).rvs()


def poisson_copula_sample_array(
    parameters: dict, x: np.array, groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
    G = parameters["beta"].shape[1]
    u = np.zeros((x.shape[0], G))
    u = glm.gaussian_copula_pseudo_obs(x.shape[0], G, parameters["covariance"], groups)

    # invert using poisson margins
    mu = np.exp(x @ parameters["beta"])
    return poisson(mu).ppf(u)


poisson_sample = glm.glm_sample_factory(
    poisson_regression_sample_array, lambda parameters: parameters["beta"].columns
)

poisson_copula_sample = glm.gaussian_copula_sample_factory(
    poisson_copula_sample_array,
    lambda parameters: parameters["beta"].columns,
)
