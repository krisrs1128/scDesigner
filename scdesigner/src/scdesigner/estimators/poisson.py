from . import gaussian_copula_factory as gcf
from . import glm_factory as factory
from .. import data
from .. import format
from anndata import AnnData
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
    # validation for param unwrapper
    series = pd.Series(params.cpu().detach().numpy())
    series.to_csv('data/poi.csv', index=False, header=False)
    
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


def poisson_regression(
    adata: AnnData,
    formula: str,
    chunk_size: int = int(1e4),
    batch_size: int = 512,
    **kwargs
) -> dict:
    loader = data.formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    result = poisson_regression_array(loader, **kwargs)
    result["parameters"] = format_poisson_parameters(result["parameters"], list(adata.var_names), list(loader.dataset.x_names))
    return result


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
