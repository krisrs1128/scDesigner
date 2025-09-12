from . import gaussian_copula_factory as gcf
from . import glm_factory as factory
from .. import format
from .. import data
from anndata import AnnData
from scipy.stats import nbinom
import numpy as np
import pandas as pd
import torch
from typing import Union

###############################################################################
## Regression functions that operate on numpy arrays
###############################################################################


def negbin_regression_likelihood(params, X_dict, y):    
    n_mean_features = X_dict["mean"].shape[1]
    n_dispersion_features = X_dict["dispersion"].shape[1]
    n_outcomes = y.shape[1]

    # form the mean and dispersion parameters
    coef_mean = params[: n_mean_features * n_outcomes].\
        reshape(n_mean_features, n_outcomes)
    coef_dispersion = params[n_mean_features * n_outcomes :].\
        reshape(n_dispersion_features, n_outcomes)
    r = torch.exp(X_dict["dispersion"] @ coef_dispersion)
    mu = torch.exp(X_dict["mean"] @ coef_mean)

    # compute the negative log likelihood
    log_likelihood = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1)
        + r * torch.log(r)
        + y * torch.log(mu)
        - (r + y) * torch.log(r + mu)
    )
    
    return -torch.sum(log_likelihood)


def negbin_initializer(x_dict, y, device):
    n_mean_features = x_dict["mean"].shape[1]
    n_outcomes = y.shape[1]
    n_dispersion_features = x_dict["dispersion"].shape[1]
    return torch.zeros(
        n_mean_features * n_outcomes\
            + n_dispersion_features * n_outcomes, 
        requires_grad=True, device=device
    )


def negbin_postprocessor(params, x_dict, y):
    n_mean_features = x_dict["mean"].shape[1]
    n_outcomes = y.shape[1]
    n_dispersion_features = x_dict["dispersion"].shape[1]
    coef_mean = format.to_np(params[:n_mean_features * n_outcomes]).\
        reshape(n_mean_features, n_outcomes)
    coef_dispersion = format.to_np(params[n_mean_features * n_outcomes:]).\
        reshape(n_dispersion_features, n_outcomes)
    return {"coef_mean": coef_mean, "coef_dispersion": coef_dispersion}


negbin_regression_array = factory.multiple_formula_regression_factory(
    negbin_regression_likelihood, negbin_initializer, negbin_postprocessor
)


###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################

def format_negbin_parameters(
    parameters: dict, var_names: list, mean_coef_index: list, 
    dispersion_coef_index: list
) -> dict:
    parameters["coef_mean"] = pd.DataFrame(
        parameters["coef_mean"], columns=var_names, index=mean_coef_index
    )
    parameters["coef_dispersion"] = pd.DataFrame(
        parameters["coef_dispersion"], columns=var_names, index=dispersion_coef_index
    )
    return parameters

def format_negbin_parameters_with_loaders(
    parameters: dict, var_names: list, dls: dict
) -> dict:
    # Extract the coefficient indices from the dataloaders
    mean_coef_index = dls["mean"].dataset.x_names
    dispersion_coef_index = dls["dispersion"].dataset.x_names
    
    return format_negbin_parameters(parameters, var_names, mean_coef_index, dispersion_coef_index)

def negbin_regression(
    adata: AnnData, formula: Union[str, dict], chunk_size: int = int(1e4), batch_size=512, **kwargs
) -> dict:
    formula = data.standardize_formula(formula, allowed_keys=['mean', 'dispersion'])
    
    loaders = data.multiple_formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    parameters = negbin_regression_array(loaders, **kwargs)
    return format_negbin_parameters(
        parameters, list(adata.var_names), loaders["mean"].dataset.x_names, loaders["dispersion"].dataset.x_names
    )

###############################################################################
## Copula versions for negative binomial regression
###############################################################################


def negbin_uniformizer(parameters, X_dict, y, epsilon=1e-3):
    r = np.exp(X_dict["dispersion"] @ parameters["coef_dispersion"])
    mu = np.exp(X_dict["mean"] @ parameters["coef_mean"])
    u1 = nbinom(n=r, p=r / (r + mu)).cdf(y)
    u2 = np.where(y > 0, nbinom(n=r, p=r / (r + mu)).cdf(y - 1), 0)
    v = np.random.uniform(size=y.shape)
    return np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)


negbin_copula_array = gcf.gaussian_copula_array_factory(
    negbin_regression_array, negbin_uniformizer
) # should accept a dictionary of dataloaders

negbin_copula = gcf.gaussian_copula_factory(
    negbin_copula_array, format_negbin_parameters_with_loaders, 
    param_name=['mean', 'dispersion']
)
