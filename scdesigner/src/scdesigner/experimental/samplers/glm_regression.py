import numpy as np
import pandas as pd
import anndata as ad
from ..estimators.glm_regression import group_indices
from scipy.stats import nbinom, norm
from formulaic import model_matrix


def negative_binomial_regression_sample_array(
    parameters: dict, x: np.array
) -> np.array:
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    r = np.repeat(r, mu.shape[0], axis=0)
    return nbinom(n=r, p=r / (r + mu)).rvs()


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


def negative_binomial_copula_sample_array(
    parameters: dict, x: np.array, groups: dict
) -> np.array:
    # initialize uniformized gaussian samples
    G = parameters["coefficient"].shape[1]
    u = np.zeros((x.shape[0], G))

    # cycle across groups
    for group, ix in groups.items():
        if type(parameters["covariance"]) is not dict:
            parameters["covariance"] = {group: parameters["covariance"]}

        z = np.random.multivariate_normal(
            mean=np.zeros(G), cov=parameters["covariance"][group], size=len(ix)
        )
        normal_distn = norm(0, np.diag(parameters["covariance"][group] ** 0.5))
        u[ix] = normal_distn.cdf(z)

    # invert using negative binomial margins
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    r = np.repeat(r, mu.shape[0], axis=0)
    return nbinom(n=r, p=r / (r + mu)).ppf(u)


def negative_binomial_copula_sample(
    parameters: dict, obs: pd.DataFrame, formula="~ 1", formula_copula="~ 1"
) -> ad.AnnData:
    x = model_matrix(formula, obs)
    groups = group_indices(formula_copula, obs)

    samples = negative_binomial_copula_sample_array(parameters, x, groups)
    result = ad.AnnData(X=samples, obs=obs)
    result.var_names = parameters["dispersion"].columns
    return result
