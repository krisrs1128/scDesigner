from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
import numpy as np


# computes PNMF weight and score, ncol specify the number of clusters
def pnmf(log_data, ncol=3):  # data is np array, log transformed read data
    U = left_singular(log_data, ncol)
    W = PNMF_EucDist(log_data, U, tol=1e-4, maxIter=100, zerotol=1e-10)
    W = W / np.linalg.norm(W, ord=2)
    S = np.dot(W.T, log_data)
    return W, S


# generate one-hot encoding for score classes
def class_generator(score, n_clusters=3):
    kmeans = KMeans(n_clusters, random_state=0)  # Specify the number of clusters
    kmeans.fit(score.T)
    labels = kmeans.labels_
    num_classes = len(np.unique(labels))
    one_hot = np.eye(num_classes)[labels].astype(int)
    return one_hot


###############################################################################
## Helpers for deriving PNMF
###############################################################################


def PNMF_EucDist(X, W_init, tol=1e-3, maxIter=500, verboseN=False, zerotol=1e-10):
    # initialization
    W = W_init  # initial W is the PCA of X
    XX = np.dot(X, X.T)

    # iterations
    for iter in range(maxIter):
        if verboseN and (iter + 1) % 10 == 0:
            print("%d iterations used." % (iter + 1))
        W_old = W

        XXW = np.dot(XX, W)
        SclFactor = np.dot(W, np.dot(W.T, XXW)) + np.dot(XXW, np.dot(W.T, W))

        # QuotientLB
        SclFactor = MatFindlb(SclFactor, zerotol)
        SclFactor = np.divide(XXW, SclFactor)
        W = np.multiply(W, SclFactor)

        norm_W = np.linalg.norm(W)
        W /= norm_W
        W = MatFind(W, zerotol)

        diffW = np.linalg.norm(W_old - W) / np.linalg.norm(W_old)
        if diffW < tol:
            break

    return W


# left singular vector of X
def left_singular(X, k):
    # X_mean = np.mean(X, axis=0)
    # X_centered = X - X_mean
    U, S, Vt = svds(X, k=k)
    return np.abs(U)


def MatFindlb(A, lb):
    B = np.ones(A.shape) * lb
    Alb = np.where(A < lb, B, A)
    return Alb


def MatFind(A, ZeroThres):
    B = np.zeros(A.shape)
    Atrunc = np.where(A < ZeroThres, B, A)
    return Atrunc
