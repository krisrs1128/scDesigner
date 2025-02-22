from anndata import AnnData
from copy import deepcopy
from formulaic import model_matrix
from scipy.stats import nbinom, norm
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch


def negative_binomial_regression_likelihood(params, X, y):
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


def check_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def to_np(x):
    return x.detach().cpu().numpy()


def negative_binomial_regression_array(
    x: np.array, y: np.array, batch_size: int = 512, lr: float = 0.1, epochs: int = 40
) -> dict:
    """
    A minimal NB regression model

    # simulate data
    n_samples, n_features, n_outcomes = 1000, 2, 4
    x_sim = np.random.normal(size=(n_samples, n_features))
    beta_sim = np.random.normal(size=(n_features, n_outcomes))
    mu_sim = np.exp(x_sim @ beta_sim)
    r_sim = np.random.uniform(.5, 1.5, n_outcomes)
    y_sim = np.random.negative_binomial(r_sim, r_sim / (r_sim + mu_sim))

    # estimate model
    negative_binomial_regression(x_sim, y_sim)
    """

    device = check_device()
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    n_features, n_outcomes = x.shape[1], y.shape[1]
    params = torch.zeros(
        n_features * n_outcomes + n_outcomes, requires_grad=True, device=device
    )
    optimizer = torch.optim.Adam([params], lr=lr)

    for _ in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = negative_binomial_regression_likelihood(params, x_batch, y_batch)
            loss.backward()
            optimizer.step()

    b_elem = n_features * n_outcomes
    beta = to_np(params[:b_elem]).reshape(n_features, n_outcomes)
    dispersion = np.exp(to_np(params[b_elem:]))
    return {"coefficient": beta, "dispersion": dispersion}


def negative_binomial_copula_array(
    x: np.array, y: np.array, groups: dict, batch_size: int = 512, 
    lr: float = 0.1, epochs: int = 20,
) -> dict:
    """
    A minimal NB copula model

    # simulate data
    n_samples, n_features, n_outcomes = 1000, 2, 4
    x_sim = np.random.normal(size=(n_samples, n_features))
    beta_sim = np.random.normal(size=(n_features, n_outcomes))
    mu_sim = np.exp(x_sim @ beta_sim)
    r_sim = np.random.uniform(.5, 1.5, n_outcomes)
    y_sim = np.random.negative_binomial(r_sim, r_sim / (r_sim + mu_sim))
    y_sim[:, 1] = y_sim[:, 0]

    # estimate model
    negative_binomial_copula(x_sim, y_sim)
    """
    # get predicted mean and dispersions
    parameters = negative_binomial_regression_array(x, y, batch_size, lr, epochs)
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    nb_distn = nbinom(n=r, p=r / (r + mu))

    # gaussianize and estimate covariance
    alpha = np.random.uniform(size=y.shape)
    u = clip(alpha * nb_distn.cdf(y) + (1 - alpha) * nb_distn.cdf(1 + y))
    parameters["covariance"] = copula_covariance(u, groups)
    return parameters


def copula_covariance(u: np.array, groups: dict):
    result = {}
    for group, ix in groups.items():
        result[group] = np.cov(norm().ppf(u[ix]).T)
 
    if len(result) == 1:
        return list(result.values())[0]
    return result


def clip(u: np.array, min: float = 1e-5, max: float = 1 - 1e-5) -> np.array:
    u[u < min] = min
    u[u > max] = max
    return u


def negative_binomial_regression(adata: AnnData, formula: str, **kwargs) -> dict:
    adata = format_input_anndata(adata)
    x = model_matrix(formula, adata.obs)
    parameters = negative_binomial_regression_array(np.array(x), adata.X, **kwargs)
    return format_nb_parameters(parameters, list(adata.var_names), list(x.columns))


def negative_binomial_copula(adata: AnnData, formula: str = "~ 1", formula_copula: str = "~ 1", **kwargs) -> dict:
    adata = format_input_anndata(adata)
    x = model_matrix(formula, adata.obs)

    groups = group_indices(formula_copula, adata.obs)
    parameters = negative_binomial_copula_array(np.array(x), adata.X, groups, **kwargs)
    parameters = format_nb_parameters(parameters, list(adata.var_names), list(x.columns))
    parameters["covariance"] = format_copula_parameters(parameters, list(adata.var_names))
    return parameters

###############################################################################
## Helpers for transforming input and output data
###############################################################################

def format_input_anndata(adata: AnnData)->AnnData:
    result = adata.copy()
    if isinstance(result.X, scipy.sparse._csc.csc_matrix):
        result.X = result.X.todense()
    return result


def format_nb_parameters(parameters: dict, var_names: list, coef_index: list) -> dict:
    parameters["coefficient"] = pd.DataFrame(
        parameters["coefficient"], columns=var_names, index=coef_index
    )
    parameters["dispersion"] = pd.DataFrame(
        parameters["dispersion"].reshape(1, -1), columns=var_names, index=["dispersion"]
    )
    return parameters

def format_copula_parameters(parameters: dict, var_names: list):
    covariance = parameters["covariance"]
    if type(covariance) is not dict:
        covariance = pd.DataFrame(
            parameters["covariance"],
            columns=list(var_names),
            index=list(var_names)
        )
    else:
        for group in covariance.keys():
            covariance[group] = pd.DataFrame(
                parameters["covariance"][group],
                columns=list(var_names),
                index=list(var_names)
            )
    return covariance

def group_indices(formula: str, obs: pd.DataFrame) -> dict:
    group_matrix = model_matrix(formula, obs)
    result = {}

    for group in group_matrix.columns:
        result[group] = np.where(group_matrix[group].values == 1)[0]
    return result