import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from formulaic import model_matrix
from . import glm_regression as glm
from . import glm_regression_factory as factory

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
    return -torch.sum(log_likelihood)


def negbin_initializer(x, y, device):
    n_features, n_outcomes = x.shape[1], y.shape[1]
    return torch.zeros(
        n_features * n_outcomes + n_outcomes, requires_grad=True, device=device
    )


def negbin_postprocessor(params, n_features, n_outcomes):
    b_elem = n_features * n_outcomes
    beta = glm.to_np(params[:b_elem]).reshape(n_features, n_outcomes)
    dispersion = np.exp(glm.to_np(params[b_elem:]))
    return {"coefficient": beta, "dispersion": dispersion}


negbin_regression_array2 = factory.glm_regression_generator(
    negbin_regression_likelihood, negbin_initializer, negbin_postprocessor
)


###############################################################################
## Regression functions that operate on AnnData objects
###############################################################################


def format_nb_parameters(parameters: dict, var_names: list, coef_index: list) -> dict:
    parameters["coefficient"] = pd.DataFrame(
        parameters["coefficient"], columns=var_names, index=coef_index
    )
    parameters["dispersion"] = pd.DataFrame(
        parameters["dispersion"].reshape(1, -1), columns=var_names, index=["dispersion"]
    )
    return parameters


def negbin_regression2(adata: AnnData, formula: str, **kwargs) -> dict:
    adata = glm.format_input_anndata(adata)
    x = model_matrix(formula, adata.obs)
    parameters = negbin_regression_array2(np.array(x), adata.X, **kwargs)
    return glm.format_nb_parameters(parameters, list(adata.var_names), list(x.columns))
