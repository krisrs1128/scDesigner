from . import docstrings as ds
from anndata import AnnData
from formulaic import model_matrix
from scipy.stats import nbinom, norm
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


@ds.doc(ds.negative_binomial_regression)
def negative_binomial_regression_array(
    x: np.array, y: np.array, batch_size: int = 512, lr: float = 0.1, epochs: int = 100
) -> dict:

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


@ds.doc(ds.negative_binomial_copula)
def negative_binomial_copula_array(
    x: np.array, y: np.array, batch_size: int = 512, lr: float = 0.1, epochs: int = 100
) -> dict:
    # get predicted mean and dispersions
    parameters = negative_binomial_regression_array(x, y, batch_size, lr, epochs)
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    nb_distn = nbinom(n=r, p=r / (r + mu))

    # gaussianize and estimate covariance
    alpha = np.random.uniform(size=y.shape)
    u = clip(alpha * nb_distn.cdf(y) + (1 - alpha) * nb_distn.cdf(1 + y))
    parameters["covariance"] = np.cov(norm().ppf(u).T)
    return parameters


def clip(u: np.array, min: float = 1e-5, max: float = 1 - 1e-5) -> np.array:
    u[u < min] = min
    u[u > max] = max
    return u


def format_nb_parameters(parameters: dict, var_names: list, coef_index: list) -> dict:
    parameters["coefficient"] = pd.DataFrame(
        parameters["coefficient"], columns=var_names, index=coef_index
    )
    parameters["dispersion"] = pd.DataFrame(
        parameters["dispersion"], columns=["dispersion"], index=var_names
    )
    return parameters


def negative_binomial_regression(adata: AnnData, formula: str, **kwargs) -> dict:
    x = model_matrix(formula, adata.obs)
    y = adata.X.todense()
    parameters = negative_binomial_regression_array(np.array(x), y, **kwargs)
    return format_nb_parameters(parameters, list(adata.var_names), list(x.columns))


def negative_binomial_copula(adata: AnnData, formula: str, **kwargs) -> dict:
    x = model_matrix(formula, adata.obs)
    y = adata.X.todense()
    parameters = negative_binomial_copula_array(np.array(x), y, **kwargs)
    parameters = format_nb_parameters(
        parameters, list(adata.var_names), list(x.columns)
    )
    parameters["covariance"] = pd.DataFrame(
        parameters["covariance"],
        columns=list(adata.var_names),
        index=list(adata.var_names),
    )
    return parameters
