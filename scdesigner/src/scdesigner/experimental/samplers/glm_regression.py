import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import nbinom
from formulaic import model_matrix


def negative_binomial_regression_sample_array(
    parameters: dict, x: np.array
) -> np.array:
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    r = np.repeat(r, mu.shape[0], axis=0)
    nb_distn = nbinom(n=r, p=r / (r + mu))
    return nb_distn.rvs()


def negative_binomial_regression_sample(
    parameters: dict, obs: pd.DataFrame, formula=None
) -> ad.AnnData:
    if formula is not None:
        x = model_matrix(formula, obs)
    else:
        x = obs

    samples = negative_binomial_regression_sample_array(parameters, x)
    result = ad.AnnData(X=samples, obs=obs)
    result.var_names = parameters["dispersion"].columns
    return result
