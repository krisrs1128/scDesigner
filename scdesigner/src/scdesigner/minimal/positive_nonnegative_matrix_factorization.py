from .formula import standardize_formula
from .loader import _to_numpy
from .simulator import Simulator
from anndata import AnnData
from formulaic import model_matrix
from scipy.stats import gamma
from typing import Union, Dict
import numpy as np
import pandas as pd
import torch

################################################################################
## Functions for estimating PNMF regression
################################################################################

# computes PNMF weight and score, ncol specify the number of clusters
def pnmf(log_data, nbase=3, **kwargs):  # data is np array, log transformed read data
    """
    Computes PNMF weight and score.

    :log_data: log transformed np array of read data
    :ncol: specify the number of clusters
    :return: W (weights, gene x base) and S (scores, base x cell) as numpy arrays
    """
    U = left_singular(log_data, nbase)
    W = pnmf_eucdist(log_data, U, **kwargs)
    W = W / np.linalg.norm(W, ord=2)
    S = W.T @ log_data
    return W, S


def gamma_regression_array(
    x: np.array, y: np.array, lr: float = 0.1, epochs: int = 40
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

    a, loc, beta = _to_numpy(a, loc, beta)
    a = a.reshape(n_features, n_outcomes)
    loc = loc.reshape(n_features, n_outcomes)
    beta = beta.reshape(n_features, n_outcomes)
    return {"a": a, "loc": loc, "beta": beta}


def class_generator(score, n_clusters=3):
    """
    Generates one-hot encoding for score classes
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters, random_state=0)  # Specify the number of clusters
    kmeans.fit(score.T)
    labels = kmeans.labels_
    num_classes = len(np.unique(labels))
    one_hot = np.eye(num_classes)[labels].astype(int)
    return labels


###############################################################################
## Helpers for deriving PNMF
###############################################################################


def pnmf_eucdist(X, W_init, maxIter=500, threshold=1e-4, tol=1e-10, verbose=False, **kwargs):
    # initialization
    W = W_init  # initial W is the PCA of X
    XX = X @ X.T

    # iterations
    for iter in range(maxIter):
        if verbose and (iter + 1) % 10 == 0:
            print("%d iterations used." % (iter + 1))
        W_old = W

        XXW = XX @ W
        SclFactor = np.dot(W, W.T @ XXW) + np.dot(XXW, W.T @ W)

        # QuotientLB
        SclFactor = MatFindlb(SclFactor, tol)
        SclFactor = XXW / SclFactor
        W = W * SclFactor  # somehow W *= SclFactor doesn't work?

        norm_W = np.linalg.norm(W)
        W /= norm_W
        W = MatFind(W, tol)

        diffW = np.linalg.norm(W_old - W) / np.linalg.norm(W_old)
        if diffW < threshold:
            break

    return W


# left singular vector of X
def left_singular(X, k):
    from scipy.sparse.linalg import svds
    U, _, _ = svds(X, k=k)
    return np.abs(U)


def MatFindlb(A, lb):
    B = np.ones(A.shape) * lb
    Alb = np.where(A < lb, B, A)
    return Alb


def MatFind(A, ZeroThres):
    B = np.zeros(A.shape)
    Atrunc = np.where(A < ZeroThres, B, A)
    return Atrunc


###############################################################################
## Helpers for training PNMF regression
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
    return shifted_gamma_pdf(y, X @ a, X @ beta, X @ loc)

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


################################################################################
## Associated PNMF Objects
################################################################################

class PositiveNMF(Simulator):
    """Positive nonnegative matrix factorization marginal estimator"""
    def __init__(self, formula: Union[Dict, str], **kwargs):
        self.formula = standardize_formula(formula, allowed_keys=['mean'])
        self.parameters = None
        self.hyperparams = kwargs


    def setup_data(self, adata: AnnData, **kwargs):
        self.log_data = np.log1p(adata.X).T
        self.n_outcomes = self.log_data.shape[1]
        self.template = adata
        self.x = model_matrix(self.formula["mean"], adata.obs)
        self.columns = self.x.columns
        self.x = np.asarray(self.x)


    def fit(self, adata: AnnData, lr: float=0.1):
        self.setup_data(adata)
        W, S = pnmf(self.log_data, **self.hyperparams)
        parameters = gamma_regression_array(self.x, S.T, lr)
        parameters["W"] = W
        self.parameters = format_gamma_parameters(
            parameters, list(self.template.var_names), list(self.columns)
        )


    def predict(self, obs=None, **kwargs):
        """Predict from an obs dataframe"""
        if obs is None:
            obs = self.template.obs

        x = model_matrix(self.formula["mean"], obs)
        a, loc, beta = (
            x @ np.exp(self.parameters["a"]),
            x @ self.parameters["loc"],
            x @ np.exp(self.parameters["beta"]),
        )
        return {"a": a, "loc": loc, "beta": beta}


    def sample(self, obs=None):
        """Generate samples."""
        if obs is None:
            obs = self.template.obs
        W = self.parameters["W"]
        parameters = self.predict(obs)
        a, loc, beta = parameters["a"], parameters["loc"], parameters["beta"]
        sim_score = gamma(a, loc, 1 / beta).rvs()
        samples = np.exp(W @ sim_score.T).T

        # thresholding samples
        floor = np.floor(samples)
        samples = floor + np.where(samples - floor < 0.9, 0, 1) - 1
        samples = np.where(samples < 0, 0, samples)

        result = AnnData(X=samples, obs=obs)
        result.var_names = self.template.var_names
        return result