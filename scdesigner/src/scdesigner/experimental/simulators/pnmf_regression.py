from anndata import AnnData
from formulaic import model_matrix
from scipy.stats import gamma
from ..estimators.pnmf import pnmf
import numpy as np
import pandas as pd
import torch, scipy


class PNMFRegressionSimulator:
    def __init__(self):  # default input: cell x gene
        self.var_names = None
        self.formula = None

    def estimate(self, adata, formula: str, nbase=20, **kwargs):
        adata = format_input_anndata(adata)
        self.var_names = adata.var_names
        self.formula = formula
        log_data = np.log1p(adata.X).T
        W, S = pnmf(log_data, nbase)
        adata = AnnData(X=S.T, obs=adata.obs)

        x = model_matrix(formula, adata.obs)
        parameters = gamma_regression_array(np.array(x), adata.X, **kwargs)
        parameters["W"] = W
        return format_gamma_parameters(
            parameters, list(self.var_names), list(x.columns)
        )

    def sample(self, parameters: dict, obs: pd.DataFrame, formula=None) -> AnnData:
        if formula is not None:
            x = model_matrix(formula, obs)
        else:
            x = obs

        W = parameters["W"]
        params = self.predict(parameters, obs, formula)
        a, loc, beta = params["a"], params["loc"], params["beta"]
        sim_score = gamma(a, loc, 1 / beta).rvs()
        samples = np.exp(W @ sim_score.T).T

        # thresholding samples
        floor = np.floor(samples)
        samples = floor + np.where(samples - floor < 0.9, 0, 1) - 1
        samples = np.where(samples < 0, 0, samples)

        result = AnnData(X=samples, obs=obs)
        result.var_names = self.var_names
        return result

    def predict(self, parameters: dict, obs: pd.DataFrame, formula=None) -> dict:
        if formula is not None:
            x = model_matrix(formula, obs)
        else:
            x = obs
        a, loc, beta = (
            x @ np.exp(parameters["a"]),
            x @ parameters["loc"],
            x @ np.exp(parameters["beta"]),
        )
        return {"a": a, "loc": loc, "beta": beta}

    def __repr__(self):
        return f"""method: 'PNMFRegression'
    formula: '{self.formula}'
    parameters: 'a', 'loc', 'beta', 'W'"""


def gamma_regression_array(
    x: np.array, y: np.array, batch_size: int = 512, lr: float = 0.1, epochs: int = 40
) -> dict:
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    n_features, n_outcomes = x.shape[1], y.shape[1]
    a = torch.zeros(n_features * n_outcomes, requires_grad=True)
    loc = torch.zeros(n_features * n_outcomes, requires_grad=True)
    beta = torch.zeros(n_features * n_outcomes, requires_grad=True)
    optimizer = torch.optim.Adam([a, loc, beta], lr=lr)

    for i in range(epochs):
        optimizer.zero_grad()
        loss = negative_gamma_log_likelihood(a, beta, loc, x, y)
        loss.backward()
        optimizer.step()

    a = to_np(a).reshape(n_features, n_outcomes)
    loc = to_np(loc).reshape(n_features, n_outcomes)
    beta = to_np(beta).reshape(n_features, n_outcomes)
    return {"a": a, "loc": loc, "beta": beta}


def gamma_regression_sample_array(parameters: dict, x: np.array) -> np.array:
    a, loc, beta = (
        x @ np.exp(parameters["a"]),
        x @ parameters["loc"],
        x @ np.exp(parameters["beta"]),
    )
    return gamma(a, loc, 1 / beta).rvs()


###############################################################################
## Helpers for fitting & sampling gamma distribution
###############################################################################


def shifted_gamma_pdf(x, alpha, beta, loc):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    mask = x < loc
    y_clamped = torch.clamp(x - loc, min=1e-12)

    log_pdf = (
        alpha * torch.log(beta)
        - torch.lgamma(alpha)
        + (alpha - 1) * torch.log(y_clamped)
        - beta * y_clamped
    )
    loss = -torch.mean(log_pdf[~mask])
    n_invalid = mask.sum()
    if n_invalid > 0:  # force samples to be greater than loc
        loss = loss + 1e10 * n_invalid.float()
    return loss


def negative_gamma_log_likelihood(log_a, log_beta, loc, X, y):
    n_features = X.shape[1]
    n_outcomes = y.shape[1]

    a = torch.exp(log_a.reshape(n_features, n_outcomes))
    beta = torch.exp(log_beta.reshape(n_features, n_outcomes))
    loc = loc.reshape(n_features, n_outcomes)
    loss = shifted_gamma_pdf(y, X @ a, X @ beta, X @ loc)
    return loss


def format_input_anndata(adata: AnnData) -> AnnData:
    result = adata.copy()
    if isinstance(result.X, scipy.sparse._csc.csc_matrix):
        result.X = result.X.todense()
    return result


def to_np(x):
    return x.detach().cpu().numpy()


def format_gamma_parameters(
    parameters: dict,
    W_index: list,
    coef_index: list,
) -> dict:
    parameters["a"] = pd.DataFrame(parameters["a"], index=coef_index)
    parameters["loc"] = pd.DataFrame(parameters["loc"], index=coef_index)
    parameters["beta"] = pd.DataFrame(parameters["beta"], index=coef_index)
    parameters["W"] = pd.DataFrame(parameters["W"], index=W_index)
    return parameters
