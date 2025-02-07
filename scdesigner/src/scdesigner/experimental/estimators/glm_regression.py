import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import nbinom, norm
from . import docstrings as ds

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


@ds.doc(ds.negative_binomial_regression)
def negative_binomial_regression(
    x: np.array, y: np.array, batch_size: int = 512, lr: float = 0.1, epochs: int = 100
) -> dict:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    n_features, n_outcomes = x.shape[1], y.shape[1]
    params = torch.zeros(n_features * n_outcomes + n_outcomes, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=lr)

    for _ in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = negative_binomial_regression_likelihood(params, x_batch, y_batch)
            loss.backward()
            optimizer.step()

    b_elem = n_features * n_outcomes
    beta = params[:b_elem].detach().numpy().reshape(n_features, n_outcomes)
    dispersion = np.exp(params[b_elem:].detach().numpy())
    return {"beta": beta, "dispersion": dispersion}


@ds.doc(ds.negative_binomial_copula)
def negative_binomial_copula(
    x: np.array, y: np.array, batch_size: int = 512, lr: float = 0.1, epochs: int = 100
) -> dict:
    # get predicted mean and dispersions
    parameters = negative_binomial_regression(x, y, batch_size, lr, epochs)
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["beta"])
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
