from anndata import AnnData
from .. import format
from .. import data
from . import glm_factory as factory
import pandas as pd
import torch


def zero_inflated_poisson_regression_likelihood(params, X, y):
    # get appropriate parameter shape
    mean_n_features = X['mean'].shape[1]
    zero_inflation_n_features = X['zero_inflation'].shape[1]
    n_outcomes = y.shape[1]

    # define the likelihood parameters
    b_elem = mean_n_features * n_outcomes
    coef_mean = params[:b_elem].reshape(mean_n_features, n_outcomes)
    coef_zero_inflation = params[b_elem:].reshape(zero_inflation_n_features, n_outcomes)

    zero_inflation = torch.sigmoid(X['zero_inflation'] @ coef_zero_inflation)
    mu = torch.exp(X['mean'] @ coef_mean)
    poisson_loglikelihood = y * torch.log(mu + 1e-10) - mu - torch.lgamma(y + 1)

    # return the mixture, with an offset to prevent log(0)
    log_likelihood = torch.log(
        zero_inflation * (y == 0) + (1 - zero_inflation) * torch.exp(poisson_loglikelihood) + 1e-10
    )
    return -torch.sum(log_likelihood)


def zero_inflated_poisson_initializer(x, y, device):
    mean_n_features = x['mean'].shape[1]
    zero_inflation_n_features = x['zero_inflation'].shape[1]
    n_outcomes = y.shape[1]
    return torch.zeros(
        mean_n_features * n_outcomes + zero_inflation_n_features * n_outcomes, requires_grad=True, device=device
    )


def zero_inflated_poisson_postprocessor(params, x, y):
    mean_n_features = x['mean'].shape[1]
    zero_inflation_n_features = x['zero_inflation'].shape[1]
    n_outcomes = y.shape[1]
    b_elem = mean_n_features * n_outcomes
    coef_mean = format.to_np(params[:b_elem]).reshape(mean_n_features, n_outcomes)
    coef_zero_inflation = format.to_np(params[b_elem:]).reshape(zero_inflation_n_features, n_outcomes)
    return {"coef_mean": coef_mean, "coef_zero_inflation": coef_zero_inflation}


zero_inflated_poisson_regression_array = factory.multiple_formula_regression_factory(
    zero_inflated_poisson_regression_likelihood,
    zero_inflated_poisson_initializer,
    zero_inflated_poisson_postprocessor,
)


###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_zero_inflated_poisson_parameters(
    parameters: dict, var_names: list, mean_coef_index: list, 
    zero_inflation_coef_index: list
) -> dict:
    parameters["coef_mean"] = pd.DataFrame(
        parameters["coef_mean"], columns=var_names, index=mean_coef_index
    )
    parameters["coef_zero_inflation"] = pd.DataFrame(
        parameters["coef_zero_inflation"], columns=var_names, index=zero_inflation_coef_index
    )
    return parameters


def zero_inflated_poisson_regression(
    adata: AnnData, formula: str, chunk_size: int = int(1e4), batch_size=512, **kwargs
) -> dict:
    formula = data.standardize_formula(formula, allowed_keys=['mean', 'zero_inflation'])
    loaders = data.multiple_formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    parameters = zero_inflated_poisson_regression_array(loaders, **kwargs)
    return format_zero_inflated_poisson_parameters(
        parameters, list(adata.var_names), loaders["mean"].dataset.x_names, loaders["zero_inflation"].dataset.x_names
    )