from scipy.stats import poisson, bernoulli
from . import glm_factory as glm
import numpy as np


def zero_inflated_poisson_sample_array(local_parameters: dict) -> np.array:
    mu, pi = (
        local_parameters["mean"],
        local_parameters["zero_inflation"],
    )
    return poisson(mu).rvs() * bernoulli(1 - pi).rvs()


zero_inflated_poisson_sample = glm.glm_sample_factory(
    zero_inflated_poisson_sample_array
)
