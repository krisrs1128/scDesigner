import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def negative_binomial_regression_likelihood(params, X, y):
    beta = params[:-1]
    log_r = params[-1]
    r = torch.exp(log_r)
    mu = torch.exp(X @ beta)
    log_likelihood = torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1) + r * torch.log(r) + y * torch.log(mu) - (r + y) * torch.log(r + mu)
    return -torch.sum(log_likelihood)


def negative_binomial_regression(X, y, batch_size=512, lr=0.1, epochs=100):
    """
    A very minimal NB regression model (single response)


    Examples
    --------
    n_samples = 1000
    n_features = 2

    X_sim = np.random.normal(size=(n_samples, n_features))
    beta_sim = np.random.normal((n_features, 1))
    mu_sim = np.exp(X_sim @ beta_sim)
    r_sim = 5
    y_sim = np.random.negative_binomial(r_sim, r_sim / (r_sim + mu_sim))

    params = negative_binomial_regression(X_sim, y_sim)
    print("True:", beta_sim, r_sim)
    print("Estimated:", params["beta"], params["dispersion"])
    """
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    n_features = X.shape[1]
    params = torch.zeros(n_features + 1, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=lr)
    
    for _ in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = negative_binomial_regression_likelihood(params, X_batch, y_batch)
            loss.backward()
            optimizer.step()
    
    return {"beta": params[:n_features].detach().numpy(), "dispersion": np.exp(params[-1].detach().numpy())}
