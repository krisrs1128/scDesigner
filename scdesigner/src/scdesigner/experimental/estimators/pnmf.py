from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
import numpy as np


# computes PNMF weight and score, ncol specify the number of clusters
def pnmf(log_data, ncol=3, **kwargs):  # data is np array, log transformed read data
    """
    Computes PNMF weight and score.

    :log_data: log transformed np array of read data
    :ncol: specify the number of clusters
    :return: W (weights, gene x base) and S (scores, base x cell) as numpy arrays
    """
    U = left_singular(log_data, ncol)
    W = pnmf_eucdist(log_data, U, **kwargs)
    W = W / np.linalg.norm(W, ord=2)
    S = W.T @ log_data
    return W, S


def class_generator(score, n_clusters=3):
    """
    Generates one-hot encoding for score classes
    """
    kmeans = KMeans(n_clusters, random_state=0)  # Specify the number of clusters
    kmeans.fit(score.T)
    labels = kmeans.labels_
    num_classes = len(np.unique(labels))
    one_hot = np.eye(num_classes)[labels].astype(int)
    return one_hot


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
