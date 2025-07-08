from anndata import AnnData
from .. import format
from .. import data
from . import glm_factory as factory
import pandas as pd
import torch


def zero_inflated_poisson_regression_likelihood(params, X, y):
    # get appropriate parameter shape
    n_features = X.shape[1]
    n_outcomes = y.shape[1]

    # define the likelihood parameters
    b_elem = n_features * n_outcomes
    beta = params[:b_elem].reshape(n_features, n_outcomes)
    logit_pi = params[b_elem:]
    pi = torch.sigmoid(logit_pi)

    mu = torch.exp(X @ beta)
    poisson_loglikelihood = y * torch.log(mu) - mu - torch.lgamma(y)

    # return the mixture, with an offset to prevent log(0)
    log_likelihood = torch.log(
        pi * (y == 0) + (1 - pi) * torch.exp(poisson_loglikelihood) + 1e-10
    )
    return -torch.sum(log_likelihood)


def zero_inflated_poisson_initializer(x, y, device):
    n_features, n_outcomes = x.shape[1], y.shape[1]
    return torch.zeros(
        n_features * n_outcomes + n_outcomes, requires_grad=True, device=device
    )


def zero_inflated_poisson_postprocessor(params, n_features, n_outcomes):
    # validation for param unwrapper
    series = pd.Series(params.cpu().detach().numpy())
    series.to_csv('data/zipoi.csv', index=False, header=False)
    
    b_elem = n_features * n_outcomes
    beta = format.to_np(params[:b_elem]).reshape(n_features, n_outcomes)
    pi = format.to_np(torch.sigmoid(params[b_elem:]))
    return {"beta": beta, "pi": pi}


zero_inflated_poisson_regression_array = factory.glm_regression_factory(
    zero_inflated_poisson_regression_likelihood,
    zero_inflated_poisson_initializer,
    zero_inflated_poisson_postprocessor,
)

###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_zero_inflated_poisson_parameters(
    parameters: dict, var_names: list, coef_index: list
) -> dict:
    parameters["beta"] = pd.DataFrame(
        parameters["beta"], columns=var_names, index=coef_index
    )
    parameters["pi"] = pd.DataFrame(
        parameters["pi"].reshape(1, -1), columns=var_names, index=["pi"]
    )
    return parameters


def zero_inflated_poisson_regression(
    adata: AnnData, formula: str, chunk_size: int = int(1e4), batch_size=512, **kwargs
) -> dict:
    loader = data.formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    parameters = zero_inflated_poisson_regression_array(loader, **kwargs)
    return format_zero_inflated_poisson_parameters(
        parameters, list(adata.var_names), list(loader.dataset.x_names)
    )
