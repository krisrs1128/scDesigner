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


def poisson_regression_likelihood(params, X, y, epsilon=1e-6):
    # get appropriate parameter shape
    n_features = X['mean'].shape[1]
    n_outcomes = y.shape[1]

    # compute the negative log likelihood
    beta = params.reshape(n_features, n_outcomes)
    mu = torch.exp(X['mean'] @ beta)
    log_likelihood = y * torch.log(mu+epsilon) - mu - torch.lgamma(y + 1)
    return -torch.sum(log_likelihood)


def poisson_initializer(x, y, device):
    n_features, n_outcomes = x['mean'].shape[1], y.shape[1]
    return torch.zeros(n_features * n_outcomes, requires_grad=True, device=device)


def poisson_postprocessor(params, x, y):
    coef_mean = format.to_np(params).reshape(x['mean'].shape[1], y.shape[1])
    return {"coef_mean": coef_mean}


poisson_regression_array = factory.multiple_formula_regression_factory(
    poisson_regression_likelihood, poisson_initializer, poisson_postprocessor
)

###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_poisson_parameters(
    parameters: dict, var_names: list, coef_index: list
) -> dict:
    parameters["coef_mean"] = pd.DataFrame(
        parameters["coef_mean"], columns=var_names, index=coef_index
    )
    return parameters


def poisson_regression(
    adata: AnnData,
    formula: str,
    chunk_size: int = int(1e4),
    batch_size: int = 512,
    **kwargs
) -> dict:
    formula = data.standardize_formula(formula, allowed_keys=['mean'])
    loaders = data.multiple_formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )

    parameters = poisson_regression_array(loaders, **kwargs)
    return format_poisson_parameters(
        parameters, list(adata.var_names), loaders["mean"].dataset.x_names
    )


###############################################################################
## Copula versions for poisson regression
###############################################################################


def poisson_uniformizer(parameters, x, y, epsilon=1e-3):
    mu = np.exp(x['mean'] @ parameters["coef_mean"])
    u1 = poisson(mu).cdf(y)
    u2 = np.where(y > 0, poisson(mu).cdf(y - 1), 0)
    v = np.random.uniform(size=y.shape)
    return np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)

def format_poisson_parameters_with_loaders(parameters: dict, var_names: list, dls: dict) -> dict:
    beta_coef_index = dls["mean"].dataset.x_names
    
    parameters["coef_mean"] = pd.DataFrame(
        parameters["coef_mean"], columns=var_names, index=beta_coef_index
    )
    return parameters

poisson_copula_array = gcf.gaussian_copula_array_factory(
    poisson_regression_array, poisson_uniformizer
) 

poisson_copula = gcf.gaussian_copula_factory(
    poisson_copula_array, format_poisson_parameters_with_loaders, ['mean']
)
