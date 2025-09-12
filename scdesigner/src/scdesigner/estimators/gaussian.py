from . import gaussian_copula_factory as gcf
from . import glm_factory as factory
from .. import format
from .. import data
from anndata import AnnData
from scipy.stats import norm
import numpy as np
import pandas as pd
import torch
from typing import Union

###############################################################################
## Regression functions that operate on numpy arrays
###############################################################################



# Gaussian regression likelihood: regression onto mean and sdev
def gaussian_regression_likelihood(params, X_dict, y):
    n_mean_features = X_dict["mean"].shape[1]
    n_sdev_features = X_dict["sdev"].shape[1]
    n_outcomes = y.shape[1]

    coef_mean = params[: n_mean_features * n_outcomes].reshape(n_mean_features, n_outcomes)
    coef_sdev = params[n_mean_features * n_outcomes :].reshape(n_sdev_features, n_outcomes)
    mu = X_dict["mean"] @ coef_mean
    sigma = torch.exp(X_dict["sdev"] @ coef_sdev)

    # Negative log likelihood for Gaussian
    log_likelihood = -0.5 * (torch.log(2 * torch.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))
    return torch.sum(log_likelihood)



def gaussian_initializer(x_dict, y, device):
    n_mean_features = x_dict["mean"].shape[1]
    n_outcomes = y.shape[1]
    n_sdev_features = x_dict["sdev"].shape[1]
    return torch.zeros(
        n_mean_features * n_outcomes + n_sdev_features * n_outcomes,
        requires_grad=True, device=device
    )



def gaussian_postprocessor(params, x_dict, y):
    n_mean_features = x_dict["mean"].shape[1]
    n_outcomes = y.shape[1]
    n_sdev_features = x_dict["sdev"].shape[1]
    coef_mean = format.to_np(params[:n_mean_features * n_outcomes]).reshape(n_mean_features, n_outcomes)
    coef_sdev = format.to_np(params[n_mean_features * n_outcomes:]).reshape(n_sdev_features, n_outcomes)
    return {"coef_mean": coef_mean, "coef_sdev": coef_sdev}



gaussian_regression_array = factory.multiple_formula_regression_factory(
    gaussian_regression_likelihood, gaussian_initializer, gaussian_postprocessor
)


###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_gaussian_parameters(
    parameters: dict, var_names: list, mean_coef_index: list, sdev_coef_index: list
) -> dict:
    parameters["coef_mean"] = pd.DataFrame(
        parameters["coef_mean"], columns=var_names, index=mean_coef_index
    )
    parameters["coef_sdev"] = pd.DataFrame(
        parameters["coef_sdev"], columns=var_names, index=sdev_coef_index
    )
    return parameters


def format_gaussian_parameters_with_loaders(
    parameters: dict, var_names: list, dls: dict
) -> dict:
    mean_coef_index = dls["mean"].dataset.x_names
    sdev_coef_index = dls["sdev"].dataset.x_names
    return format_gaussian_parameters(
        parameters, var_names, mean_coef_index, sdev_coef_index
    )


def gaussian_regression(
    adata: AnnData, formula: Union[str, dict], chunk_size: int = int(1e4), 
    batch_size=512, **kwargs
) -> dict:
    formula = data.standardize_formula(formula, allowed_keys=['mean', 'sdev'])
    loaders = data.multiple_formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    parameters = gaussian_regression_array(loaders, **kwargs)
    return format_gaussian_parameters(
        parameters, list(adata.var_names), loaders["mean"].dataset.x_names, 
        loaders["sdev"].dataset.x_names
    )


###############################################################################
## Copula versions for gaussian regression
###############################################################################

def gaussian_uniformizer(parameters, X_dict, y, epsilon=1e-3):
    mu = X_dict["mean"] @ parameters["coef_mean"]
    sigma = np.exp(X_dict["sdev"] @ parameters["coef_sdev"])
    u = norm.cdf(y, loc=mu, scale=sigma)
    u = np.clip(u, epsilon, 1 - epsilon)
    return u

gaussian_copula_array = gcf.gaussian_copula_array_factory(
    gaussian_regression_array, gaussian_uniformizer
)

gaussian_copula = gcf.gaussian_copula_factory(
    gaussian_copula_array, format_gaussian_parameters_with_loaders,
    param_name=['mean', 'sdev']
)
