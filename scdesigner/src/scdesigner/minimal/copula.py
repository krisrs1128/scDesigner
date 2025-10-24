from typing import Dict, Callable, Tuple
import torch
from anndata import AnnData
from .loader import adata_loader
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional
class Copula(ABC):
    def __init__(self, formula: str, **kwargs):
        self.formula = formula
        self.loader = None
        self.n_outcomes = None
        self.parameters = None # Should be a dictionary of CovarianceStructure objects

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

    # @abstractmethod
    # def format_parameters(self):
    #     raise NotImplementedError
    
class CovarianceStructure:
    """
    Data structure to efficiently store and access covariance information for fast copula sampling.
    
    Attributes:
    -----------
    cov : np.ndarray
        Full covariance matrix for modeled genes, shape (n_modeled_genes, n_modeled_genes)
    modeled_names : pd.Index
        Names of the modeled genes
    modeled_indices : np.ndarray
        Indices of the modeled genes in the original gene ordering
    remaining_var : np.ndarray
        Diagonal variances for remaining genes, shape (n_remaining_genes,)
    remaining_indices : np.ndarray
        Indices of the remaining genes in the original gene ordering. If None, all remaining genes are assumed to be the last n_remaining_genes genes.
    remaining_names : np.ndarray
        Names of the remaining genes
    """
    
    def __init__(self, cov: np.ndarray, 
                 modeled_names: pd.Index, 
                 modeled_indices: Optional[np.ndarray] = None,
                 remaining_var: Optional[np.ndarray] = None, 
                 remaining_indices: Optional[np.ndarray] = None, 
                 remaining_names: Optional[pd.Index] = None):
        
        self.cov = pd.DataFrame(cov, index=modeled_names, columns=modeled_names)
        
        if modeled_indices is not None:
            self.modeled_indices = modeled_indices
        else:
            self.modeled_indices = np.arange(len(modeled_names))
        
        if remaining_var is not None:
            self.remaining_var = pd.Series(remaining_var, index=remaining_names)
        else: 
            self.remaining_var = None
        
        self.remaining_indices = remaining_indices
        self.num_modeled_genes = len(modeled_names)
        self.num_remaining_genes = len(remaining_indices) if remaining_indices is not None else 0
        self.total_genes = self.num_modeled_genes + self.num_remaining_genes
        
    def __repr__(self):
        if self.remaining_var is None:
            return self.cov.__repr__()
        else:
            return f"CovarianceStructure(modeled_genes={self.num_modeled_genes}, \
                total_genes={self.total_genes})"
    
    def _repr_html_(self):
        """Jupyter Notebook display"""
        if self.remaining_var is None:
            return self.cov._repr_html_()
        else:
            html = f"<b>CovarianceStructure:</b> {self.num_modeled_genes} modeled genes, {self.total_genes} total<br>"
            html += "<h4>Modeled Covariance Matrix</h4>" + self.cov._repr_html_()
            html += "<h4>Remaining Gene Variances</h4>" + self.remaining_var.to_frame("variance").T._repr_html_()
            return html
    
    def decorrelate(self, row_pattern: str, col_pattern: str):
        """Decorrelate the covariance matrix for the given row and column patterns.
        """
        from .transform import data_frame_mask
        m1 = data_frame_mask(self.cov, ".", col_pattern)
        m2 = data_frame_mask(self.cov, row_pattern, ".")
        mask = (m1 | m2)
        np.fill_diagonal(mask, False)
        self.cov.values[mask] = 0
        
    def correlate(self, row_pattern: str, col_pattern: str, factor: float):
        """Multiply selected off-diagonal entries by factor.

        Args:
            row_pattern (str): The regex pattern for the row names to match.
            col_pattern (str): The regex pattern for the column names to match.
            factor (float): The factor to multiply the off-diagonal entries by.
        """
        from .transform import data_frame_mask
        m1 = data_frame_mask(self.cov, ".", col_pattern)
        m2 = data_frame_mask(self.cov, row_pattern, ".")
        mask = (m1 | m2)
        np.fill_diagonal(mask, False)
        self.cov.values[mask] = self.cov.values[mask] * factor
    
    @property
    def shape(self):
        return (self.total_genes, self.total_genes)
    
    def to_full_matrix(self):
        """
        Convert to full covariance matrix for compatibility/debugging.
        Returns:
        --------
        np.ndarray : Full covariance matrix with shape (total_genes, total_genes)
        """
        if self.remaining_var is None:
            return self.cov.values
        else:
            full_cov = np.zeros((self.total_genes, self.total_genes))
            
            # Fill in top-k block
            ix_modeled = np.ix_(self.modeled_indices, self.modeled_indices)
            full_cov[ix_modeled] = self.cov.values
            
            # Fill in diagonal for remaining genes
            full_cov[self.remaining_indices, self.remaining_indices] = self.remaining_var.values
        
        return full_cov 