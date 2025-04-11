from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import torch


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


def class_generator(score, n_clusters=3):
    """
    Generates one-hot encoding for score classes
    """
    kmeans = KMeans(n_clusters, random_state=0)  # Specify the number of clusters
    kmeans.fit(score.T)
    labels = kmeans.labels_
    num_classes = len(np.unique(labels))
    one_hot = np.eye(num_classes)[labels].astype(int)
    return labels


###############################################################################
## Helpers for deriving PNMF
###############################################################################


def pnmf_eucdist(X, W_init, maxIter=500, threshold=1e-4, tol=1e-10, verbose=False):
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
