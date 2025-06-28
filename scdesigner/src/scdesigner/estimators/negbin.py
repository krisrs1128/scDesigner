import warnings
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
    num_mean_features = X_dict["mean"].shape[1]
    num_dispersion_features = X_dict["dispersion"].shape[1]
    n_outcomes = y.shape[1]

    # form the mean and dispersion parameters
    beta_mean = params[: num_mean_features * n_outcomes].\
        reshape(num_mean_features, n_outcomes)
    beta_dispersion = params[num_mean_features * n_outcomes :].\
        reshape(num_dispersion_features, n_outcomes)
    r, mu = torch.exp(X_dict["dispersion"] @ beta_dispersion), \
        torch.exp(X_dict["mean"] @ beta_mean)

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
    num_mean_features = x_dict["mean"].shape[1]
    num_outcomes = y.shape[1]
    num_dispersion_features = x_dict["dispersion"].shape[1]
    return torch.zeros(
        num_mean_features * num_outcomes\
            + num_dispersion_features * num_outcomes, 
        requires_grad=True, device=device
    )


def negbin_postprocessor(params, x_dict, y):
    num_mean_features = x_dict["mean"].shape[1]
    num_outcomes = y.shape[1]
    num_dispersion_features = x_dict["dispersion"].shape[1]
    beta_mean = format.to_np(params[:num_mean_features * num_outcomes]).\
        reshape(num_mean_features, num_outcomes)
    beta_dispersion = format.to_np(params[num_mean_features * num_outcomes:]).\
        reshape(num_dispersion_features, num_outcomes)
    return {"beta": beta_mean, "gamma": beta_dispersion}


negbin_regression_array = factory.multiple_formula_regression_factory(
    negbin_regression_likelihood, negbin_initializer, negbin_postprocessor
)


###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################

def format_negbin_parameters(
    parameters: dict, var_names: list, mean_coef_index: list, dispersion_coef_index: list
) -> dict:
    parameters["beta"] = pd.DataFrame(
        parameters["beta"], columns=var_names, index=mean_coef_index
    )
    parameters["gamma"] = pd.DataFrame(
        parameters["gamma"], columns=var_names, index=dispersion_coef_index
    )
    return parameters


def negbin_regression(
    adata: AnnData, formula: Union[str, dict], chunk_size: int = int(1e4), batch_size=512, **kwargs
) -> dict:
    """
    formula: Union[str, dict]
    if formula is a string, it is the formula for the mean parameter
    if formula is a dictionary, it is a dictionary of formulas for the mean and dispersion parameters
    if a dictionary is provided, it should have "mean" key, 
        and if "dispersion" key is not provided, it is set to "~ 1"
    """
    
    if isinstance(formula, str):
        formula = {'mean': formula, 'dispersion': '~ 1'}
    elif not isinstance(formula, dict):
        raise ValueError("formula must be a string or a dictionary")
    
    allowed_keys = ['mean', 'dispersion']
    extra_keys = [key for key in formula if key not in allowed_keys]

    if extra_keys:
        warnings.warn(
            f"There are unused formulas in the dictionary formula for negative binomial \
                regression: {extra_keys}",
            UserWarning,
        )

    if "dispersion" not in formula:
        formula["dispersion"] = "~ 1"
    
    
    loaders = data.multiple_formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    ) # a dictionary of dataloaders for each formula
    parameters = negbin_regression_array(loaders, **kwargs)
    return format_negbin_parameters(
        parameters, list(adata.var_names), loaders["mean"].dataset.x_names, loaders["dispersion"].dataset.x_names
    )

###############################################################################
## Copula versions for negative binomial regression
###############################################################################


def negbin_uniformizer(parameters, x, y):
    r, mu = np.exp(parameters["gamma"]), np.exp(x @ parameters["beta"])
    nb_distn = nbinom(n=r, p=r / (r + mu))
    alpha = np.random.uniform(size=y.shape)
    return gcf.clip(alpha * nb_distn.cdf(y) + (1 - alpha) * nb_distn.cdf(1 + y))


negbin_copula_array = gcf.gaussian_copula_array_factory(
    negbin_regression_array, negbin_uniformizer
)

negbin_copula = gcf.gaussian_copula_factory(
    negbin_copula_array, format_negbin_parameters
)
