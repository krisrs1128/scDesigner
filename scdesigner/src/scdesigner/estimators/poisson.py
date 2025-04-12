from . import gaussian_copula_factory as gcf
from . import glm_factory as factory
from .. import format
from anndata import AnnData
from formulaic import model_matrix
from scipy.stats import poisson
import numpy as np
import pandas as pd
import torch

###############################################################################
## Regression functions that operate on numpy arrays
###############################################################################


def poisson_regression_likelihood(params, X, y):
    # get appropriate parameter shape
    n_features = X.shape[1]
    n_outcomes = y.shape[1]

    # compute the negative log likelihood
    beta = params.reshape(n_features, n_outcomes)
    mu = torch.exp(X @ beta)
    log_likelihood = y * torch.log(mu) - mu - torch.lgamma(y)
    return -torch.sum(log_likelihood)


def poisson_initializer(x, y, device):
    n_features, n_outcomes = x.shape[1], y.shape[1]
    return torch.zeros(n_features * n_outcomes, requires_grad=True, device=device)


def poisson_postprocessor(params, n_features, n_outcomes):
    beta = format.to_np(params).reshape(n_features, n_outcomes)
    return {"beta": beta}


poisson_regression_array = factory.glm_regression_factory(
    poisson_regression_likelihood, poisson_initializer, poisson_postprocessor
)

###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_poisson_parameters(
    parameters: dict, var_names: list, coef_index: list
) -> dict:
    parameters["beta"] = pd.DataFrame(
        parameters["beta"], columns=var_names, index=coef_index
    )
    return parameters


def poisson_regression(adata: AnnData, formula: str, **kwargs) -> dict:
    adata = format.format_input_anndata(adata)
    x = model_matrix(formula, adata.obs)
    parameters = poisson_regression_array(np.array(x), adata.X, **kwargs)
    return format_poisson_parameters(parameters, list(adata.var_names), list(x.columns))


###############################################################################
## Copula versions for poisson regression
###############################################################################


def poisson_uniformizer(parameters, x, y):
    mu = np.exp(x @ parameters["beta"])
    nb_distn = poisson(mu)
    alpha = np.random.uniform(size=y.shape)
    return gcf.clip(alpha * nb_distn.cdf(y) + (1 - alpha) * nb_distn.cdf(1 + y))


poisson_copula = gcf.gaussian_copula_factory(
    gcf.gaussian_copula_array_factory(poisson_regression_array, poisson_uniformizer),
    format_poisson_parameters,
)
