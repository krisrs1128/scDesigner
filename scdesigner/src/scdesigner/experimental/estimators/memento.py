from ..data import SparseMatrixLoader
from .estimators import Estimator
import numpy as np
import scipy.sparse
import torch

class MementoEstimator(Estimator):
    def __init__(self, q: float = 1):
        self.q = q

    def estimate(self,
                 loader: SparseMatrixLoader) -> dict:
        X = []
        for x in loader:
            X.append(x)
        X = vstack_sparse(X)

        cell_totals = torch.sum(X, axis=1, keepdim=True).to_dense().flatten()
        cell_totals_ = np.repeat(cell_totals[:, None], X.shape[1], axis=1)
        mu = torch.sum((1 / (cell_totals_ * self.q)) * X, axis=0, keepdim=True).to_dense().flatten()
        mu = mu / X.shape[0]

        sigma = memento_sigma(X, mu, cell_totals, self.q)
        covariance = memento_covariance(X, mu, cell_totals, sigma, self.q)
        return {"covariance": covariance, "mean": mu}

def memento_sigma(Y, mu, cell_totals, q=1):
    n_cells, n_genes = Y.shape
    cell_totals_ = np.repeat(cell_totals[:, None], n_genes, axis=1)
    term1 = (1 / ((q * cell_totals_) ** 2)) * (Y * Y)
    term1 = term1.sum(axis=0, keepdim=True).to_dense().flatten()

    term2 = (1 / ((q * cell_totals_) ** 2)) * Y * (1 - q)
    term2 = term2.sum(axis=0, keepdim=True).to_dense().flatten()

    return 1 / n_cells * (term1 - term2) - mu ** 2

def memento_covariance(Y, mu, cell_totals, sigma, q=1):
    # sparse matrix multiplication for the first term
    cell_totals_inv_sq = scipy.sparse.diags(np.array((cell_totals * q) ** -2))
    Y = torch_to_scipy(Y)
    term1 = (1 / Y.shape[0]) * Y.T @ cell_totals_inv_sq @ Y 

    # combine into dense D x D covariance
    covariance = term1.todense() - (mu[:, None] @ mu[:, None].T).numpy()
    np.fill_diagonal(covariance, sigma)
    covariance = torch.Tensor(covariance)
    covariance[covariance < 0] = 0
    return covariance

def torch_to_scipy(x):
    """
    Convert sparse CSR tensor to sparse CSR scipy array.
    """
    ix = x.to_sparse().indices()
    return scipy.sparse.csr_matrix((x.values(), ix), x.shape)

def vstack_sparse(tensor_list):
    """
    Helper to vertically concatenate sparse csr tensors. 

    Eventually it should be resolved by the pytorch team.
    https://github.com/pytorch/pytorch/issues/98861
    """
    scipy_csr_matrices = [torch_to_scipy(t) for t in tensor_list]

    scipy_csr = scipy.sparse.vstack(scipy_csr_matrices)
    return torch.sparse_csr_tensor(
        scipy_csr.indptr,
        scipy_csr.indices,
        scipy_csr.data,
        size=scipy_csr.shape
    )