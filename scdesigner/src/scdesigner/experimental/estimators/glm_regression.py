import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import nbinom, norm


def negative_binomial_regression_likelihood(params, X, y):
    n_features = X.shape[1]
    n_outcomes = y.shape[1]

    # form the mean and dispersion parameters
    beta = params[:n_features * n_outcomes].reshape(n_features, n_outcomes)
    log_r = params[n_features * n_outcomes:]
    r = torch.exp(log_r)
    mu = torch.exp(X @ beta)

    # compute the negative log likelihood
    log_likelihood = torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1) + r * torch.log(r) + y * torch.log(mu) - (r + y) * torch.log(r + mu)
    return -torch.sum(log_likelihood)


def negative_binomial_regression(x: np.array, y: np.array, batch_size:int =512, lr: float=0.1, epochs: int=100)->dict:
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
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    n_features = x.shape[1]
    n_outcomes = y.shape[1]
    params = torch.zeros(n_features * n_outcomes + n_outcomes, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=lr)
    
    for _ in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = negative_binomial_regression_likelihood(params, x_batch, y_batch)
            loss.backward()
            optimizer.step()
    
    beta = params[:n_features * n_outcomes].detach().numpy().reshape(n_features, n_outcomes)
    dispersion = np.exp(params[n_features * n_outcomes:].detach().numpy())
    return {"beta": beta, "dispersion": dispersion}


def negative_binomial_copula(x: np.array, y: np.array, batch_size: int=512, lr:
                             float=0.1, epochs: int=100) -> dict:
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
    parameters = negative_binomial_regression(x, y, batch_size, lr, epochs)
    r = np.exp(parameters["dispersion"])
    mu = np.exp(x @ parameters["beta"])
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