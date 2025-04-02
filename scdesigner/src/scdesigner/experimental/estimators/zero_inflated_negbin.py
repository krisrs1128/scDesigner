import torch
import numpy as np
import pandas as pd
from anndata import AnnData
from formulaic import model_matrix
from . import glm_regression as glm
from . import glm_regression_factory as factory

###############################################################################
## Regression functions that operate on numpy arrays
###############################################################################


def zero_inflated_negbin_regression_likelihood(params, X, y):
    # get appropriate parameter shape
    n_features = X.shape[1]
    n_outcomes = y.shape[1]

    # define the likelihood parameters
    b_elem = n_features * n_outcomes
    beta = params[:b_elem].reshape(n_features, n_outcomes)
    log_r = params[b_elem : (b_elem + n_outcomes)]
    logit_pi = params[(b_elem + n_outcomes) :]
    pi = torch.sigmoid(logit_pi)

    # negative binomial component
    r, mu = torch.exp(log_r), torch.exp(X @ beta)
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


def zero_inflated_negbin_initializer(x, y, device):
    n_features, n_outcomes = x.shape[1], y.shape[1]
    return torch.zeros(
        n_features * n_outcomes + 2 * n_outcomes, requires_grad=True, device=device
    )


def zero_inflated_negbin_postprocessor(params, n_features, n_outcomes):
    b_elem = n_features * n_outcomes
    beta = glm.to_np(params[:b_elem]).reshape(n_features, n_outcomes)
    dispersion = glm.to_np(torch.exp(params[b_elem : (b_elem + n_outcomes)]))
    pi = glm.to_np(torch.sigmoid(params[(b_elem + n_outcomes) :]))
    return {"beta": beta, "dispersion": dispersion, "pi": pi}


zero_inflated_negbin_regression_array = factory.glm_regression_generator(
    zero_inflated_negbin_regression_likelihood,
    zero_inflated_negbin_initializer,
    zero_inflated_negbin_postprocessor,
)

###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_zero_inflated_negbin_parameters(
    parameters: dict, var_names: list, coef_index: list
) -> dict:
    parameters["beta"] = pd.DataFrame(
        parameters["beta"], columns=var_names, index=coef_index
    )
    parameters["dispersion"] = pd.DataFrame(
        parameters["dispersion"].reshape(1, -1), columns=var_names, index=["dispersion"]
    )
    parameters["pi"] = pd.DataFrame(
        parameters["pi"].reshape(1, -1), columns=var_names, index=["pi"]
    )
    return parameters


def zero_inflated_negbin_regression(adata: AnnData, formula: str, **kwargs) -> dict:
    adata = glm.format_input_anndata(adata)
    x = model_matrix(formula, adata.obs)
    parameters = zero_inflated_negbin_regression_array(np.array(x), adata.X, **kwargs)
    return format_zero_inflated_negbin_parameters(
        parameters, list(adata.var_names), list(x.columns)
    )
