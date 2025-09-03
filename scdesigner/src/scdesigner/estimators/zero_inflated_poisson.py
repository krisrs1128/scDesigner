from anndata import AnnData
from .. import format
from .. import data
from . import glm_factory as factory
import pandas as pd
import torch


def zero_inflated_poisson_regression_likelihood(params, X, y):
    # get appropriate parameter shape
    beta_n_features = X['beta'].shape[1]
    pi_n_features = X['pi'].shape[1]
    n_outcomes = y.shape[1]

    # define the likelihood parameters
    b_elem = beta_n_features * n_outcomes
    coef_beta = params[:b_elem].reshape(beta_n_features, n_outcomes)
    coef_pi = params[b_elem:].reshape(pi_n_features, n_outcomes)

    pi = torch.sigmoid(X['pi'] @ coef_pi)
    mu = torch.exp(X['beta'] @ coef_beta)
    poisson_loglikelihood = y * torch.log(mu + 1e-10) - mu - torch.lgamma(y + 1)

    # return the mixture, with an offset to prevent log(0)
    log_likelihood = torch.log(
        pi * (y == 0) + (1 - pi) * torch.exp(poisson_loglikelihood) + 1e-10
    )
    return -torch.sum(log_likelihood)


def zero_inflated_poisson_initializer(x, y, device):
    beta_n_features = x['beta'].shape[1]
    pi_n_features = x['pi'].shape[1]
    n_outcomes = y.shape[1]
    return torch.zeros(
        beta_n_features * n_outcomes + pi_n_features * n_outcomes, requires_grad=True, device=device
    )


def zero_inflated_poisson_postprocessor(params, x, y):
    beta_n_features = x['beta'].shape[1]
    pi_n_features = x['pi'].shape[1]
    n_outcomes = y.shape[1]
    b_elem = beta_n_features * n_outcomes
    coef_beta = format.to_np(params[:b_elem]).reshape(beta_n_features, n_outcomes)
    coef_pi = format.to_np(params[b_elem:]).reshape(pi_n_features, n_outcomes)
    return {"coef_beta": coef_beta, "coef_pi": coef_pi}


zero_inflated_poisson_regression_array = factory.multiple_formula_regression_factory(
    zero_inflated_poisson_regression_likelihood,
    zero_inflated_poisson_initializer,
    zero_inflated_poisson_postprocessor,
)


###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_zero_inflated_poisson_parameters(
    parameters: dict, var_names: list, beta_coef_index: list, 
    pi_coef_index: list
) -> dict:
    parameters["coef_beta"] = pd.DataFrame(
        parameters["coef_beta"], columns=var_names, index=beta_coef_index
    )
    parameters["coef_pi"] = pd.DataFrame(
        parameters["coef_pi"], columns=var_names, index=pi_coef_index
    )
    return parameters


def zero_inflated_poisson_regression(
    adata: AnnData, formula: str, chunk_size: int = int(1e4), batch_size=512, **kwargs
) -> dict:
    formula = data.standardize_formula(formula, allowed_keys={'beta', 'pi'})
    loaders = data.multiple_formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    parameters = zero_inflated_poisson_regression_array(loaders, **kwargs)
    return format_zero_inflated_poisson_parameters(
        parameters, list(adata.var_names), loaders["beta"].dataset.x_names, loaders["pi"].dataset.x_names
    )