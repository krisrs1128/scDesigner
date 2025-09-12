from scipy.stats import norm
from . import glm_factory as glm
from typing import Union
import numpy as np


def gaussian_regression_sample_array(local_parameters: dict) -> np.array:
    sigma, mu = local_parameters["sdev"], local_parameters["mean"] # dataframes of shape (n, g)
    return norm(loc=mu, scale=sigma).rvs()


gaussian_regression_sample = glm.glm_sample_factory(gaussian_regression_sample_array)
