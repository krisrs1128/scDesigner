from . import gaussian_copula_factory as gcf
from .. import format
from .. import data
from . import glm_factory as factory
from anndata import AnnData
from scipy.stats import nbinom
import numpy as np
import pandas as pd
import torch

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
    # validation for param unwrapper
    series = pd.Series(params.cpu().detach().numpy())
    series.to_csv('data/zinb.csv', index=False, header=False)
    
    b_elem = n_features * n_outcomes
    beta = format.to_np(params[:b_elem]).reshape(n_features, n_outcomes)
    gamma = format.to_np(torch.exp(params[b_elem : (b_elem + n_outcomes)]))
    pi = format.to_np(torch.sigmoid(params[(b_elem + n_outcomes) :]))
    return {"beta": beta, "gamma": gamma, "pi": pi}


zero_inflated_negbin_regression_array = factory.glm_regression_factory(
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
    parameters["gamma"] = pd.DataFrame(
        parameters["gamma"].reshape(1, -1), columns=var_names, index=["gamma"]
    )
    parameters["pi"] = pd.DataFrame(
        parameters["pi"].reshape(1, -1), columns=var_names, index=["pi"]
    )
    return parameters


def zero_inflated_negbin_regression(
    adata: AnnData, formula: str, chunk_size: int = int(1e4), batch_size=512, **kwargs
) -> dict:
    loader = data.formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    result = zero_inflated_negbin_regression_array(loader, **kwargs)
    result["parameters"] = format_zero_inflated_negbin_parameters(
        result["parameters"], list(adata.var_names), list(loader.dataset.x_names)
    )
    return result


###############################################################################
## Copula versions for ZINB regression
###############################################################################


def zero_inflated_negbin_uniformizer(parameters, x, y):
    r, mu, pi = (
        np.exp(parameters["gamma"]),
        np.exp(x @ parameters["beta"]),
        parameters["pi"],
    )
    nb_distn = nbinom(n=r, p=r / (r + mu))
    alpha = np.random.uniform(size=y.shape)

    cdf1 = pi + (1 - pi) * nb_distn.cdf(y)
    cdf2 = pi + (1 - pi) * nb_distn.cdf(1 + y)
    return gcf.clip(alpha * cdf1 + (1 - alpha) * cdf2)


zero_inflated_negbin_copula = gcf.gaussian_copula_factory(
    gcf.gaussian_copula_array_factory(
        zero_inflated_negbin_regression_array, zero_inflated_negbin_uniformizer
    ),
    format_zero_inflated_negbin_parameters,
)
