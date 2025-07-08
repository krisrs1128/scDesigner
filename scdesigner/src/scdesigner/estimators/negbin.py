from . import gaussian_copula_factory as gcf
from . import glm_factory as factory
from .. import format
from .. import data
from anndata import AnnData
from scipy.stats import nbinom
import numpy as np
import pandas as pd
import torch

###############################################################################
## Regression functions that operate on numpy arrays
###############################################################################


def negbin_regression_likelihood(params, X, y):
    n_features = X.shape[1]
    n_outcomes = y.shape[1]

    # form the mean and dispersion parameters
    beta = params[: n_features * n_outcomes].reshape(n_features, n_outcomes)
    log_r = params[n_features * n_outcomes :]
    r, mu = torch.exp(log_r), torch.exp(X @ beta)

    # compute the negative log likelihood
    log_likelihood = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1)
        + r * torch.log(r)
        + y * torch.log(mu)
        - (r + y) * torch.log(r + mu)
    )
    return log_likelihood


def negbin_initializer(x, y, device):
    n_features, n_outcomes = x.shape[1], y.shape[1]
    return torch.zeros(
        n_features * n_outcomes + n_outcomes, requires_grad=True, device=device
    )


def negbin_postprocessor(params, n_features, n_outcomes):
    # validation for param unwrapper
    series = pd.Series(params.cpu().detach().numpy())
    series.to_csv('data/nbreg.csv', index=False, header=False)

    b_elem = n_features * n_outcomes
    beta = format.to_np(params[:b_elem]).reshape(n_features, n_outcomes)
    dispersion = format.to_np(params[b_elem:])
    return {"beta": beta, "gamma": dispersion}


negbin_regression_array = factory.glm_regression_factory(
    negbin_regression_likelihood, negbin_initializer, negbin_postprocessor
)


###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_negbin_parameters(
    parameters: dict, var_names: list, coef_index: list
) -> dict:
    parameters["beta"] = pd.DataFrame(
        parameters["beta"], columns=var_names, index=coef_index
    )
    parameters["gamma"] = pd.DataFrame(
        parameters["gamma"].reshape(1, -1), columns=var_names, index=["dispersion"]
    )
    return parameters


def negbin_regression(
    adata: AnnData, formula: str, chunk_size: int = int(1e4), batch_size=512, **kwargs
) -> dict:
    loader = data.formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    result = negbin_regression_array(loader, **kwargs)
    result["parameters"] = format_negbin_parameters(
        result["parameters"], list(adata.var_names), loader.dataset.x_names
    )
    return result


###############################################################################
## Copula versions for negative binomial regression
###############################################################################


# def negbin_uniformizer(parameters, x, y):
#     r, mu = np.exp(parameters["gamma"]), np.exp(x @ parameters["beta"])
#     nb_distn = nbinom(n=r, p=r / (r + mu))
#     alpha = np.random.uniform(size=y.shape)
#     return gcf.clip(alpha * nb_distn.cdf(y) + (1 - alpha) * nb_distn.cdf(1 + y))


def negbin_uniformizer(parameters, x, y, epsilon=1e-3):
    np.random.seed(42)
    r, mu = np.exp(parameters["gamma"]), np.exp(x @ parameters["beta"])
    u1 = nbinom(n=r, p=r / (r + mu)).cdf(y)
    u2 = np.where(y > 0, nbinom(n=r, p=r / (r + mu)).cdf(y - 1), 0)
    v = np.random.uniform(size=y.shape)
    # may need to add extra step for selecting partial cells?
    p = np.clip(v * u1 + (1 - v) * u2, epsilon, 1 - epsilon)
    return p


negbin_copula_array = gcf.gaussian_copula_array_factory(
    negbin_regression_array, negbin_uniformizer
)

negbin_copula = gcf.gaussian_copula_factory(
    negbin_copula_array, format_negbin_parameters
)
