import warnings
from . import gaussian_copula_factory as gcf
from .. import format
from .. import data
from . import glm_factory as factory
from anndata import AnnData
from scipy.stats import nbinom
import numpy as np
import pandas as pd
import torch
from typing import Union
from scipy.special import expit

###############################################################################
## Regression functions that operate on numpy arrays
###############################################################################


def zero_inflated_negbin_regression_likelihood(params, X_dict, y):
    # get appropriate parameter shape
    n_mean_features = X_dict["mean"].shape[1]
    n_dispersion_features = X_dict["dispersion"].shape[1]
    n_zero_inflation_features = X_dict["zero_inflation"].shape[1]
    n_outcomes = y.shape[1]

    # define the likelihood parameters
    coef_mean = params[: n_mean_features * n_outcomes].\
        reshape(n_mean_features, n_outcomes)
    coef_dispersion = params[n_mean_features * n_outcomes :\
        n_mean_features * n_outcomes + n_dispersion_features * n_outcomes].\
        reshape(n_dispersion_features, n_outcomes)
    coef_zero_inflation = params[n_mean_features * n_outcomes + \
        n_dispersion_features * n_outcomes :].\
        reshape(n_zero_inflation_features, n_outcomes)
    
    mu = torch.exp(X_dict["mean"] @ coef_mean)
    r = torch.exp(X_dict["dispersion"] @ coef_dispersion)
    pi = torch.sigmoid(X_dict["zero_inflation"] @ coef_zero_inflation)

    # negative binomial component
    negbin_loglikelihood = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1)
        + r * torch.log(r)
        + y * torch.log(mu)
        - (r + y) * torch.log(r + mu)
    )

    # return the mixture, with an offset to prevent log(0)
    log_likelihood = torch.log(
        pi * (y == 0) + (1 - pi) * torch.exp(negbin_loglikelihood) + 1e-10
    )
    return -torch.sum(log_likelihood)


def zero_inflated_negbin_initializer(X_dict, y, device):
    n_mean_features = X_dict["mean"].shape[1]
    n_dispersion_features = X_dict["dispersion"].shape[1]
    n_zero_inflation_features = X_dict["zero_inflation"].shape[1]
    n_outcomes = y.shape[1]
    return torch.zeros(
        n_mean_features * n_outcomes + n_dispersion_features * n_outcomes + \
        n_zero_inflation_features * n_outcomes, requires_grad=True, device=device
    )


def zero_inflated_negbin_postprocessor(params, X_dict, y):
    n_mean_features = X_dict["mean"].shape[1]
    n_dispersion_features = X_dict["dispersion"].shape[1]
    n_zero_inflation_features = X_dict["zero_inflation"].shape[1]
    n_outcomes = y.shape[1]
    coef_mean = format.to_np(params[:n_mean_features * n_outcomes]).\
        reshape(n_mean_features, n_outcomes)
    coef_dispersion = format.to_np(params[n_mean_features * n_outcomes\
        : n_mean_features * n_outcomes + n_dispersion_features * n_outcomes]).\
        reshape(n_dispersion_features, n_outcomes)
    coef_zero_inflation = format.to_np(params[n_mean_features * n_outcomes \
        + n_dispersion_features * n_outcomes :]).\
        reshape(n_zero_inflation_features, n_outcomes)
    return {"coef_mean": coef_mean, "coef_dispersion": coef_dispersion,\
        "coef_zero_inflation": coef_zero_inflation}


zero_inflated_negbin_regression_array = factory.multiple_formula_regression_factory(
    zero_inflated_negbin_regression_likelihood,
    zero_inflated_negbin_initializer,
    zero_inflated_negbin_postprocessor,
)

###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_zero_inflated_negbin_parameters(
    parameters: dict, var_names: list, mean_coef_index: 
        list, dispersion_coef_index: list, zero_inflation_coef_index: list
) -> dict:
    parameters["coef_mean"] = pd.DataFrame(
        parameters["coef_mean"], columns=var_names, index=mean_coef_index
    )
    parameters["coef_dispersion"] = pd.DataFrame(
        parameters["coef_dispersion"], columns=var_names, index=dispersion_coef_index
    )
    parameters["coef_zero_inflation"] = pd.DataFrame(
        parameters["coef_zero_inflation"], columns=var_names, index=zero_inflation_coef_index
    )
    return parameters

def format_zero_inflated_negbin_parameters_with_loaders(
    parameters: dict, var_names: list, dls: dict
) -> dict:
    mean_coef_index = dls["mean"].dataset.x_names
    dispersion_coef_index = dls["dispersion"].dataset.x_names
    zero_inflation_coef_index = dls["zero_inflation"].dataset.x_names
    return format_zero_inflated_negbin_parameters(
        parameters, var_names, mean_coef_index, dispersion_coef_index, zero_inflation_coef_index
    )

def standardize_zero_inflated_negbin_formula(formula: Union[str, dict]) -> dict:
    '''
    Convert string formula to dict and validate type.
    If formula is a string, it is the formula for the mean parameter.
    If formula is a dictionary, it is a dictionary of formulas for the mean, dispersion, and zero_inflation parameters.
    '''
    # Convert string formula to dict and validate type
    formula = {'mean': formula, 'dispersion': '~ 1', 'zero_inflation': '~ 1'} \
        if isinstance(formula, str) else formula
    if not isinstance(formula, dict):
        raise ValueError("formula must be a string or a dictionary")
    
    # Define allowed keys and set defaults
    allowed_keys = {'mean', 'dispersion', 'zero_inflation'}
    formula_keys = set(formula.keys())

    # check for required keys and warn about extras
    if not formula_keys & allowed_keys:
        raise ValueError("formula must have at least one of \
                the following keys: mean, dispersion, zero_inflation")
    
    # warn about unused keys
    if extra_keys := formula_keys - allowed_keys:
        warnings.warn(
            f"Invalid formulas in dictionary for zero-inflated \
                negative binomial regression: {extra_keys}",
            UserWarning,
        )
    
    # set default values for missing keys
    formula.update({k: '~ 1' for k in allowed_keys - formula_keys})
    return formula


def zero_inflated_negbin_regression(
    adata: AnnData, formula: Union[str, dict], chunk_size: int = int(1e4), batch_size=512, **kwargs
) -> dict:
    formula = data.standardize_formula(formula, allowed_keys=['mean', 'dispersion', 'zero_inflation'])
    
    loaders = data.multiple_formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )

    parameters = zero_inflated_negbin_regression_array(loaders, **kwargs)
    return format_zero_inflated_negbin_parameters(
        parameters, list(adata.var_names), loaders["mean"].dataset.x_names, 
        loaders["dispersion"].dataset.x_names, loaders["zero_inflation"].dataset.x_names
    )


###############################################################################
## Copula versions for ZINB regression
###############################################################################


def zero_inflated_negbin_uniformizer(parameters, X_dict, y, epsilon=1e-3):
    r, mu, pi = (
        np.exp(X_dict["dispersion"] @ parameters["coef_dispersion"]),
        np.exp(X_dict["mean"] @ parameters["coef_mean"]),
        expit(X_dict["zero_inflation"] @ parameters["coef_zero_inflation"]),
    )
    nb_distn = nbinom(n=r, p=r / (r + mu))
    u1 = pi + (1 - pi) * nb_distn.cdf(y)
    u2 = np.where(y > 0, pi + (1 - pi) * nb_distn.cdf(y-1), 0)
    v = np.random.uniform(size=y.shape)
    return np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)


zero_inflated_negbin_copula = gcf.gaussian_copula_factory(
    gcf.gaussian_copula_array_factory(
        zero_inflated_negbin_regression_array, zero_inflated_negbin_uniformizer
    ),
    format_zero_inflated_negbin_parameters_with_loaders,
    param_name=['mean', 'dispersion', 'zero_inflation'],
)
