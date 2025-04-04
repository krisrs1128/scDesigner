from scipy.stats import nbinom
from . import glm_factory as glm
import numpy as np


def negbin_regression_sample_array(parameters: dict, x: np.array) -> np.array:
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    r = np.repeat(r, mu.shape[0], axis=0)
    return nbinom(n=r, p=r / (r + mu)).rvs()


def negative_binomial_copula_sample_array(
    parameters: dict, x: np.array, groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
    G = parameters["coefficient"].shape[1]
    u = np.zeros((x.shape[0], G))
    u = glm.gaussian_copula_pseudo_obs(x.shape[0], G, parameters["covariance"], groups)

    # invert using negative binomial margins
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    r = np.repeat(r, mu.shape[0], axis=0)
    return nbinom(n=r, p=r / (r + mu)).ppf(u)


negbin_sample = glm.glm_sample_factory(negbin_regression_sample_array)

negbin_copula_sample = glm.gaussian_copula_sample_factory(
    negative_binomial_copula_sample_array,
    lambda parameters: parameters["dispersion"].columns,
)
