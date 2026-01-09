from typing import Dict, Callable, Tuple
import torch
from anndata import AnnData
from ..data.loader import adata_loader
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Union


class Copula(ABC):
    """Abstract Copula Class

    The scDesign3 model is built from two components: a collection of marginal
    models, and a copula to tie them together. This class implements an abstract
    version of the copula. Within this class, we may define different subclasses
    that implement various types of regularization or dependencies on
    experimental and biological conditions. Despite these differences, the
    overall class must always provide utilities for fitting and sampling
    dependent uniform variables.

    Parameters
    ----------
    formula : str
        A string describing the dependence of the copula on experimental or
        biological conditions. We support predictors for categorical variables
        like cell type; this corresponds to estimating a different covariance
        for each category.
    Attributes
    ----------
    loader : torch.utils.data.DataLoader
        A data loader object is used to estimate the covariance one batch at a
        time. This allows estimation of the covariance structure in a streaming
        way, without having to load all data into memory.
    n_outcomes : int
        The number of features modeled by this marginal model. For example,
        this corresponds to the number of genes being simulated.
    parameters : Dict[str, CovarianceStructure]
        A dictionary of CovarianceStructure objects. Each key corresponds to a
        different category specified in the original formula. The covariance
        structure stores the relationships among genes. It can be a standard
        covariance matrix, but may also use more memory-efficient approximations
        like when using CovarianceStructure with a constraint on
        num_modeled_genes.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc3k()[:, :300]
    >>>
    >>> class DummyCopula(Copula):
    ...     def fit(self):
    ...         pass
    ...     def likelihood(self):
    ...         pass
    ...     def num_params(self):
    ...         return 0
    ...     def pseudo_obs(self, x_dict):
    ...         return np.random.uniform(size=(x_dict["group"].shape[0], self.n_outcomes))
    ...
    >>> model = DummyCopula({"group": "~ 1"})
    >>> model.setup_data(adata, {"group": "~ 1"})
    >>> model.fit()
    """
    def __init__(self, formula: Union[str, dict], **kwargs):
        self.formula = formula
        self.loader = None
        self.n_outcomes = None
        self.parameters = None # Should be a dictionary of CovarianceStructure objects

    def setup_data(self, adata: AnnData, marginal_formula: Dict[str, str], batch_size: int = 1024, **kwargs):
        """
        Populate the .loader attribute

        Parameters
        ----------
        adata : AnnData
            This is the object on which we want to estimate the simulator. This
            serves as the template for all downstream fitting.

        marginal_formula : Dict[str, str]
            A dictionary or string specifying the relationship between the columns
            of an input data frame (adata.obs, adata.var, or similar attributes) and
            the parameters of the marginal model. If only a string is provided,
            then the means are allowed to depend on the design parameters, while all
            other parameters are treated as fixed. If a dictionary is provided,
            each key should correspond to a parameter. The string values should be
            in a format that can be parsed by the formulaic package.  For example,
            '~ x' will ensure that the parameter varies linearly with X.

        Returns
        -------
        None
            This method does not return anything but populates the self.adata,
            formula, loader, and n_outcomes attributes based on the provided
            adata input object.
        """
        self.adata = adata
        self.formula = self.formula | marginal_formula #
        self.loader = adata_loader(adata, self.formula, batch_size=batch_size, **kwargs)
        X_batch, _ = next(iter(self.loader))
        self.n_outcomes = X_batch.shape[1]

    def decorrelate(self, row_pattern: str, col_pattern: str, group: Union[str, list, None] = None):
        """
        Decorrelate the covariance matrix for the given row and column patterns.

        This method can be used to generate synthetic null data where particular
        pairs of features are forced to be uncorrelated with one another. Any
        indices of the covariance that lie in the intersection of the specified
        row and column patterns will be set to zero.

        Parameters
        ----------
        row_pattern : str
            The regex pattern for the row names to match.
        col_pattern : str
            The regex pattern for the column names to match.
        group : Union[str, list, None], optional
            The group or groups to apply the transformation to. If None, the
            transformation is applied to all groups.

        Returns
        -------
        None
            This method does not return anything but modifies self parameters as
            a side effect.
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
        """
        Multiply selected off-diagonal entries by factor.

        To adjust the signal strength in a power analysis, we may want to
        rescale the correlation for specific entries in the covariance matrix.
        This function is used to apply a multiplicative factor to selected
        entries, allowing targeted modification of correlation strength.

        Parameters
        ----------
        factor : float
            The factor to multiply the off-diagonal entries by.
        row_pattern : str
            The regex pattern for the row names to match.
        col_pattern : str
            The regex pattern for the column names to match.
        group : Union[str, list, None], optional
            The group or groups to apply the transformation to. If None, the
            transformation is applied to all groups.

        Returns
        -------
        None
            This method does not return anything but modifies self parameters as
            a side effect.
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
        """
        Fit a Copula

        Copula models are estimated by transforming the observed data onto the
        [0, 1] space of percentiles. See the .invert() method within class
        Marginal.

        Parameters
        ----------
        uniformizer : Callable
            Function to transform data to uniform marginals. See .invert()
            within class Marginal for an example.
        **kwargs
            Additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def pseudo_obs(self, x_dict: Dict):
        """
        Sample from a Copula

        Dependent uniform variables can be sampled from the copula conditional
        on a specific design matrix X (encoding biological and experimental
        covariates). For example, this will sample uniform variables with
        dependence reflecting the cell type specified by X.

        Parameters
        ----------
        x_dict : Dict
            A dictionary of tensors, with one key/value pair per parameter.
            These tensors are the conditioning information to pass to the
            .predict() function of this distribution class. They are the
            numerical design matrices implied by the initializing formulas.
        """
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, uniformizer: Callable, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]]):
        """
        Parameters
        ----------
        uniformizer : Callable
            Function to transform data to uniform marginals.
        batch : Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Batch of data.
        """
        raise NotImplementedError

    @abstractmethod
    def num_params(self, **kwargs):
        """
        Covariance Parameters

        This returns the number of free parameters in the overall copula model.
        This is useful for assessing model complexity.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments.
        """
        raise NotImplementedError


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> sigma = np.random.uniform(size=(5, 5))
    >>> modeled_names = ["A", "B", "C", "D", "E"]
    >>> sigma = pd.DataFrame(sigma, columns=modeled_names, index=modeled_names)
    >>> covariance = CovarianceStructure(sigma, modeled_names)
    """
    def __init__(self, cov: np.ndarray,
                 modeled_names: pd.Index,
                 modeled_indices: Optional[np.ndarray] = None,
                 remaining_var: Optional[np.ndarray] = None,
                 remaining_indices: Optional[np.ndarray] = None,
                 remaining_names: Optional[pd.Index] = None):
        """
        Initialize a CovarianceStructure object.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix for modeled genes, shape (n_modeled_genes, n_modeled_genes)
        modeled_names : pd.Index
            Gene names for the modeled genes
        modeled_indices : Optional[np.ndarray], optional
            Indices of modeled genes in original ordering. Defaults to sequential indices.
        remaining_var : Optional[np.ndarray], optional
            Diagonal variances for remaining genes, shape (n_remaining_genes,)
        remaining_indices : Optional[np.ndarray], optional
            Indices of remaining genes in original ordering
        remaining_names : Optional[pd.Index], optional
            Gene names for remaining genes
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
        """
        Jupyter Notebook display

        Returns
        -------
        str
            HTML representation of the object.
        """
        if self.remaining_var is None:
            return self.cov._repr_html_()
        else:
            html = f"<b>CovarianceStructure:</b> {self.num_modeled_genes} modeled genes, {self.total_genes} total<br>"
            html += "<h4>Modeled Covariance Matrix</h4>" + self.cov._repr_html_()
            html += "<h4>Remaining Gene Variances</h4>" + self.remaining_var.to_frame("variance").T._repr_html_()
            return html

    def decorrelate(self, row_pattern: str, col_pattern: str):
        """
        Decorrelate the covariance matrix for the given row and column patterns.

        This method can be used to generate synthetic null data where particular
        pairs of features are forced to be uncorrelated with one another. Any
        indices of the covariance that lie in the intersection of the specified
        row and column patterns will be set to zero.

        Parameters
        ----------
        row_pattern : str
            The regex pattern for the row names to match.
        col_pattern : str
            The regex pattern for the column names to match.
        """
        from ..transform.transform import data_frame_mask
        m1 = data_frame_mask(self.cov, ".", col_pattern)
        m2 = data_frame_mask(self.cov, row_pattern, ".")
        mask = (m1 | m2)
        np.fill_diagonal(mask, False)
        self.cov.values[mask] = 0

    def correlate(self, row_pattern: str, col_pattern: str, factor: float):
        """
        Multiply selected off-diagonal entries by factor.

        To adjust the signal strength in a power analysis, we may want to
        rescale the correlation for specific entries in the covariance matrix.
        This function is used to apply a multiplicative factor to selected
        entries, allowing targeted modification of correlation strength.


        Parameters
        ----------
        row_pattern : str
            The regex pattern for the row names to match.
        col_pattern : str
            The regex pattern for the column names to match.
        factor : float
            The factor to multiply the off-diagonal entries by.
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
        Convert to full covariance matrix for compatibility and debugging.

        Returns
        -------
        np.ndarray
            Full covariance matrix with shape (total_genes, total_genes)
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