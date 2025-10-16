from typing import Dict, Callable, Tuple
import torch
from anndata import AnnData
from .loader import adata_loader
from abc import ABC, abstractmethod
import numpy as np
class Copula(ABC):
    def __init__(self, formula: str, **kwargs):
        self.formula = formula
        self.loader = None
        self.n_outcomes = None
        self.parameters = None

    def setup_data(self, adata: AnnData, marginal_formula: Dict[str, str], batch_size: int = 1024, **kwargs):
        self.adata = adata
        self.formula = self.formula | marginal_formula
        self.loader = adata_loader(adata, self.formula, batch_size=batch_size, **kwargs)
        X_batch, _ = next(iter(self.loader))
        self.n_outcomes = X_batch.shape[1]

    @abstractmethod
    def fit(self, uniformizer: Callable, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def pseudo_obs(self, x_dict: Dict):
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, uniformizer: Callable, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]):
        raise NotImplementedError

    @abstractmethod
    def num_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def format_parameters(self):
        raise NotImplementedError
    
class FastCovarianceStructure:
    """
    Data structure to efficiently store and access covariance information for fast copula sampling.
    
    Attributes:
    -----------
    top_k_cov : np.ndarray
        Full covariance matrix for top-k most prevalent genes, shape (top_k, top_k)
    remaining_var : np.ndarray  
        Diagonal variances for remaining genes, shape (remaining_genes,)
    top_k_indices : np.ndarray
        Indices of the top-k genes in the original gene ordering
    remaining_indices : np.ndarray
        Indices of the remaining genes in the original gene ordering
    gene_total_expression : np.ndarray
        Total expression levels used for gene selection, shape (total_genes,)
    """
    
    def __init__(self, top_k_cov, remaining_var, top_k_indices, remaining_indices, gene_total_expression):
        self.top_k_cov = top_k_cov
        self.remaining_var = remaining_var
        self.top_k_indices = top_k_indices
        self.remaining_indices = remaining_indices
        self.gene_total_expression = gene_total_expression
        self.top_k = len(top_k_indices)
        self.total_genes = len(top_k_indices) + len(remaining_indices)
        
    def __repr__(self):
        return (f"FastCovarianceStructure(top_k={self.top_k}, "
                f"remaining_genes={len(self.remaining_indices)}, "
                f"total_genes={self.total_genes})")
    
    def to_full_matrix(self):
        """
        Convert to full covariance matrix for compatibility/debugging.
        
        Returns:
        --------
        np.ndarray : Full covariance matrix with shape (total_genes, total_genes)
        """
        full_cov = np.zeros((self.total_genes, self.total_genes))
        
        # Fill in top-k block
        ix_top = np.ix_(self.top_k_indices, self.top_k_indices)
        full_cov[ix_top] = self.top_k_cov
        
        # Fill in diagonal for remaining genes
        full_cov[self.remaining_indices, self.remaining_indices] = self.remaining_var
        
        return full_cov 