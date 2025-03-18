from anndata import AnnData
from copy import deepcopy
from formulaic import model_matrix
from scipy.stats import nbinom
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch


class NBRegression:
    def __init__(self):  # default input: cell x gene
        self.var_names = None
        self.formula = None
        self.shape = None

    def estimate(self, adata: AnnData, formula: str, **kwargs) -> dict:
        adata = format_input_anndata(adata)
        self.formula = formula
        self.shape = adata.X.shape
        x = model_matrix(formula, adata.obs)
        parameters = negative_binomial_regression_array(np.array(x), adata.X, **kwargs)
        return format_nb_parameters(parameters, list(adata.var_names), list(x.columns))

    def sample(self, parameters: dict, obs: pd.DataFrame, formula=None) -> AnnData:
        if formula is not None:
            x = model_matrix(formula, obs)
        else:
            x = obs

        r, mu = negative_binomial_regression_sample_array(parameters, x)
        samples = nbinom(n=r, p=r / (r + mu)).rvs()
        result = AnnData(X=samples, obs=obs)
        result.var_names = parameters["dispersion"].columns
        return result

    def predict(self, parameters: dict, obs: pd.DataFrame, formula=None) -> dict:
        if formula is not None:
            x = model_matrix(formula, obs)
        else:
            x = obs

        r, mu = negative_binomial_regression_sample_array(parameters, x)
        return {"coefficient": mu, "dispersion": r}

    def __str__(self):
        return f"""scDesigner object with n_obs x n_vars = {self.shape[0]} x {self.shape[1]}
    method: 'NBRegression'
    formula: '{self.formula}'
    parameters: 'coefficient', 'dispersion'"""


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


def negative_binomial_regression_sample_array(
    parameters: dict, x: np.array
) -> np.array:
    r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
    r = np.repeat(r, mu.shape[0], axis=0)
    return r, mu


###############################################################################
## Helpers for fitting & sampling NB regression
###############################################################################


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


def format_input_anndata(adata: AnnData) -> AnnData:
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


def check_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def to_np(x):
    return x.detach().cpu().numpy()
