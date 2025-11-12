from typing import Dict, Callable, Tuple
import torch
from anndata import AnnData
from ..data.loader import adata_loader
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Union


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
    
    def decorrelate(self, row_pattern: str, col_pattern: str, group: Union[str, list, None] = None):
        """Decorrelate the covariance matrix for the given row and column patterns.
        
        Args:
            row_pattern (str): The regex pattern for the row names to match.
            col_pattern (str): The regex pattern for the column names to match.
            group (Union[str, list, None]): The group or groups to apply the transformation to. If None, the transformation is applied to all groups.
        """
        if group is None:
            for g in self.groups:
                self.parameters[g].decorrelate(row_pattern, col_pattern)
        elif isinstance(group, str):
            self.parameters[group].decorrelate(row_pattern, col_pattern)
        else:
            for g in group:
                self.parameters[g].decorrelate(row_pattern, col_pattern)
                
    def correlate(self, factor: float, row_pattern: str, col_pattern: str, group: Union[str, list, None] = None):
        """Multiply selected off-diagonal entries by factor.
        
        Args:
            row_pattern (str): The regex pattern for the row names to match.
            col_pattern (str): The regex pattern for the column names to match.
            factor (float): The factor to multiply the off-diagonal entries by.
            group (Union[str, list, None]): The group or groups to apply the transformation to. If None, the transformation is applied to all groups.
        """
        if group is None:
            for g in self.groups:
                self.parameters[g].correlate(row_pattern, col_pattern, factor)
        elif isinstance(group, str):
            self.parameters[group].correlate(row_pattern, col_pattern, factor)
        else:
            for g in group:
                self.parameters[g].correlate(row_pattern, col_pattern, factor)
                
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
    Efficient storage for covariance matrices in copula-based gene expression modeling.
    
    This class provides memory-efficient storage for covariance information by storing
    either a full covariance matrix or a block matrix with diagonal variances for
    remaining genes. This enables fast copula estimation and sampling for large
    gene expression datasets.
    

    
    Attributes
    ----------
    cov : pd.DataFrame
        Covariance matrix for modeled genes with gene names as index/columns
    modeled_indices : np.ndarray
        Indices of modeled genes in original ordering
    remaining_var : pd.Series or None
        Diagonal variances for remaining genes, None if full matrix stored
    remaining_indices : np.ndarray or None
        Indices of remaining genes in original ordering
    num_modeled_genes : int
        Number of modeled genes
    num_remaining_genes : int
        Number of remaining genes (0 if full matrix stored)
    total_genes : int
        Total number of genes
    """
    
    def __init__(self, cov: np.ndarray, 
                 modeled_names: pd.Index, 
                 modeled_indices: Optional[np.ndarray] = None,
                 remaining_var: Optional[np.ndarray] = None, 
                 remaining_indices: Optional[np.ndarray] = None, 
                 remaining_names: Optional[pd.Index] = None):
        """initialize a CovarianceStructure object.

        Args:
            cov (np.ndarray): Covariance matrix for modeled genes, shape (n_modeled_genes, n_modeled_genes)
            modeled_names (pd.Index): Gene names for the modeled genes
            modeled_indices (Optional[np.ndarray], optional): Indices of modeled genes in original ordering. Defaults to sequential indices.
            remaining_var (Optional[np.ndarray], optional): Diagonal variances for remaining genes, shape (n_remaining_genes,)
            remaining_indices (Optional[np.ndarray], optional): Indices of remaining genes in original ordering
            remaining_names (Optional[pd.Index], optional): Gene names for remaining genes
        """
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
        from ..transform.transform import data_frame_mask
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
        from ..transform.transform import data_frame_mask
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