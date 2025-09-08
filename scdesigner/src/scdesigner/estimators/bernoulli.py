import pandas as pd
from . import gaussian_copula_factory as gcf
from . import glm_factory as factory
from .. import data
from .. import format
from . import poisson as poi
from anndata import AnnData
from scipy.stats import bernoulli
import numpy as np
import torch
from typing import Union

###############################################################################
## Regression functions that operate on numpy arrays
###############################################################################


def bernoulli_regression_likelihood(params, X_dict, y):
    # get appropriate parameter shape
    n_features = X_dict["mean"].shape[1]
    n_outcomes = y.shape[1]

    # compute the negative log likelihood
    coef_mean = params.reshape(n_features, n_outcomes)
    theta = torch.sigmoid(X_dict["mean"] @ coef_mean)
    log_likelihood = y * torch.log(theta) + (1 - y) * torch.log(1 - theta)
    return -torch.sum(log_likelihood)

def bernoulli_initializer(X_dict, y, device):
    n_features = X_dict["mean"].shape[1]
    n_outcomes = y.shape[1]
    return torch.zeros(n_features * n_outcomes, requires_grad=True, device=device)

def bernoulli_postprocessor(params, X_dict, y):
    coef_mean = format.to_np(params).reshape(X_dict["mean"].shape[1], y.shape[1])
    return {"coef_mean": coef_mean}

bernoulli_regression_array = factory.multiple_formula_regression_factory(
    bernoulli_regression_likelihood, bernoulli_initializer, bernoulli_postprocessor
)

###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def bernoulli_regression(
    adata: AnnData, formula: Union[str, dict], chunk_size: int = int(1e4), batch_size=512, **kwargs
) -> dict:
    formula = data.standardize_formula(formula, allowed_keys=['mean'])
    loaders = data.multiple_formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    parameters = bernoulli_regression_array(loaders, **kwargs)
    return poi.format_poisson_parameters(
        parameters, list(adata.var_names), list(loaders["mean"].dataset.x_names)
    )


###############################################################################
## Copula versions for bernoulli regression
###############################################################################


def bernoulli_uniformizer(parameters, X_dict, y, epsilon=1e-3, random_seed=42):
    np.random.seed(random_seed)
    theta = torch.sigmoid(torch.from_numpy(X_dict["mean"] @ parameters["coef_mean"])).numpy()
    u1 =  bernoulli(theta).cdf(y)
    u2 = np.where(y > 0,  bernoulli(theta).cdf(y - 1), 0)
    v = np.random.uniform(size=y.shape)
    return np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)

def format_bernoulli_parameters_with_loaders(parameters: dict, var_names: list, dls: dict) -> dict:
    coef_mean_index = dls["mean"].dataset.x_names
    
    parameters["coef_mean"] = pd.DataFrame(
        parameters["coef_mean"], columns=var_names, index=coef_mean_index
    )
    return parameters

bernoulli_copula = gcf.gaussian_copula_factory(
    gcf.gaussian_copula_array_factory(bernoulli_regression_array, bernoulli_uniformizer),
    format_bernoulli_parameters_with_loaders,
    param_name=['mean']
)
