from ..data import stack_collate, multiple_formula_group_loader
from .. import data
from anndata import AnnData
from collections.abc import Callable
from typing import Union
from scipy.stats import norm
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

###############################################################################
## General copula factory functions
###############################################################################


def gaussian_copula_array_factory(marginal_model: Callable, uniformizer: Callable):
    def copula_fun(loaders: dict[str, DataLoader], lr: float = 0.1, epochs: int = 40, **kwargs):
        # for the marginal model, ignore the groupings
        # Strip all dataloaders and create a dictionary to pass to marginal_model
        formula_loaders = {}
        for key in loaders.keys():
            formula_loaders[key] = strip_dataloader(loaders[key], pop="Stack" in type(loaders[key].dataset).__name__)
        
        # Call marginal_model with the dictionary of stripped dataloaders
        parameters = marginal_model(formula_loaders, lr=lr, epochs=epochs, **kwargs)

        # estimate covariance, allowing for different groups
        parameters["covariance"] = copula_covariance(parameters, loaders, uniformizer)
        return parameters

    return copula_fun


def fast_gaussian_copula_array_factory(marginal_model: Callable, uniformizer: Callable, top_k: int):
    """
    Factory function for fast Gaussian copula array computation using top-k gene modeling.

    """
    def copula_fun(loaders: dict[str, DataLoader], lr: float = 0.1, epochs: int = 40, **kwargs):
        # for the marginal model, ignore the groupings
        # Strip all dataloaders and create a dictionary to pass to marginal_model
        formula_loaders = {}
        for key in loaders.keys():
            formula_loaders[key] = strip_dataloader(loaders[key], pop="Stack" in type(loaders[key].dataset).__name__)
        
        # Call marginal_model with the dictionary of stripped dataloaders
        parameters = marginal_model(formula_loaders, lr=lr, epochs=epochs, **kwargs)

        # estimate covariance using fast method, allowing for different groups
        parameters["covariance"] = fast_copula_covariance(parameters, loaders, uniformizer, top_k)
        return parameters

    return copula_fun


def gaussian_copula_factory(copula_array_fun: Callable, 
                            parameter_formatter: Callable, 
                            param_name: list = None):
    def copula_fun(
        adata: AnnData,
        formula: Union[str, dict] = "~ 1",
        grouping_var: str = None,
        chunk_size: int = int(1e4),
        batch_size: int = 512,
        **kwargs
    ) -> dict:  
        
        if param_name is not None:
            formula = data.standardize_formula(formula, param_name)
        
        dls = multiple_formula_group_loader(
            adata,
            formula,
            grouping_var,
            chunk_size=chunk_size,
            batch_size=batch_size,
        ) # returns a dictionary of dataloaders
        parameters = copula_array_fun(dls, **kwargs)
        
        # Pass the full dls to parameter_formatter so it can extract what it needs
        parameters = parameter_formatter(
            parameters, adata.var_names, dls
        )
        parameters["covariance"] = format_copula_parameters(parameters, adata.var_names)
        return parameters

    return copula_fun




def copula_covariance(parameters: dict, loaders: dict[str, DataLoader], uniformizer: Callable):
    
    first_loader = next(iter(loaders.values()))
    D = next(iter(first_loader))[1].shape[1] #dimension of y
    groups = first_loader.dataset.groups # a list of strings of group names
    sums = {g: np.zeros(D) for g in groups}
    second_moments = {g: np.eye(D) for g in groups}
    Ng = {g: 0 for g in groups}
    keys = list(loaders.keys())
    loaders = list(loaders.values())
    num_keys = len(keys)
    
    for batches in zip(*loaders):
        x_batch_dict = {
            keys[i]: batches[i][0].cpu().numpy() for i in range(num_keys)
        }
        y_batch = batches[0][1].cpu().numpy()
        memberships = batches[0][2] # should be identical for all keys
        
        u = uniformizer(parameters, x_batch_dict, y_batch)
        for g in groups:
            ix = np.where(np.array(memberships) == g)
            z = norm().ppf(u[ix])
            second_moments[g] += z.T @ z
            sums[g] += z.sum(axis=0)
            Ng[g] += len(ix[0])

    result = {}
    for g in groups:
        mean = sums[g] / Ng[g]
        result[g] = second_moments[g] / Ng[g] - np.outer(mean, mean)

    if len(groups) == 1:
        return list(result.values())[0] 
    return result


def fast_copula_covariance(parameters: dict, loaders: dict[str, DataLoader], uniformizer: Callable, top_k: int):
    """
    Compute an efficient approximation of copula covariance by modeling only the top-k most prevalent genes
    with full covariance and approximating the rest with diagonal covariance.
    
    Parameters:
    -----------
    parameters : dict
        Model parameters dictionary
    loaders : dict[str, DataLoader]
        Dictionary of data loaders
    uniformizer : Callable
        Function to convert to uniform distribution
    top_k : int
        Number of top genes to model with full covariance
        
    Returns:
    --------
    dict or FastCovarianceStructure:
        - If single group: FastCovarianceStructure containing:
          * top_k_cov: (top_k, top_k) full covariance matrix for top genes
          * remaining_var: (remaining_genes,) diagonal variances for remaining genes  
          * top_k_indices: indices of top-k genes
          * remaining_indices: indices of remaining genes
          * gene_total_expression: total expression levels for gene selection
        - If multiple groups: dict mapping group names to FastCovarianceStructure objects
    """
    
    first_loader = next(iter(loaders.values()))
    D = next(iter(first_loader))[1].shape[1] #dimension of y
    groups = first_loader.dataset.groups # a list of strings of group names
    
    # Validate top_k parameter
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if top_k >= D:
        # If top_k is larger than total genes, fall back to regular covariance
        return copula_covariance(parameters, loaders, uniformizer)
    
    # Step 1: Calculate total expression for each gene to determine prevalence
    gene_total_expression = np.zeros(D)
    
    keys = list(loaders.keys())
    loaders_list = list(loaders.values())
    num_keys = len(keys)
    
    # Calculate total expression across all batches
    for batches in zip(*loaders_list):
        y_batch = batches[0][1].cpu().numpy()
        gene_total_expression += y_batch.sum(axis=0)
    
    # Step 2: Select top-k most prevalent genes
    top_k_indices = np.argsort(gene_total_expression)[-top_k:]
    remaining_indices = np.argsort(gene_total_expression)[:-top_k]
    
    # Step 3: Compute statistics for both top-k and remaining genes
    sums_top_k = {g: np.zeros(top_k) for g in groups}
    second_moments_top_k = {g: np.zeros((top_k, top_k)) for g in groups}
    
    sums_remaining = {g: np.zeros(len(remaining_indices)) for g in groups}
    second_moments_remaining = {g: np.zeros(len(remaining_indices)) for g in groups}
    
    Ng = {g: 0 for g in groups}
    
    # Reset loaders for second pass
    loaders_list = list(loaders.values())
    
    for batches in zip(*loaders_list):
        x_batch_dict = {
            keys[i]: batches[i][0].cpu().numpy() for i in range(num_keys)
        }
        y_batch = batches[0][1].cpu().numpy()
        memberships = batches[0][2] # should be identical for all keys
        
        u = uniformizer(parameters, x_batch_dict, y_batch)
        
        for g in groups:
            ix = np.where(np.array(memberships) == g)
            if len(ix[0]) == 0:
                continue
                
            z = norm().ppf(u[ix])
            
            # Process top-k genes with full covariance
            z_top_k = z[:, top_k_indices]
            second_moments_top_k[g] += z_top_k.T @ z_top_k
            sums_top_k[g] += z_top_k.sum(axis=0)
            
            # Process remaining genes with diagonal covariance only
            z_remaining = z[:, remaining_indices]
            second_moments_remaining[g] += (z_remaining ** 2).sum(axis=0)
            sums_remaining[g] += z_remaining.sum(axis=0)
            
            Ng[g] += len(ix[0])

    # Step 4: Compute final covariance structures
    result = {}
    for g in groups:
        if Ng[g] == 0:
            continue
            
        # Full covariance for top-k genes
        mean_top_k = sums_top_k[g] / Ng[g]
        cov_top_k = second_moments_top_k[g] / Ng[g] - np.outer(mean_top_k, mean_top_k)
        
        # Diagonal variance for remaining genes
        mean_remaining = sums_remaining[g] / Ng[g]
        var_remaining = second_moments_remaining[g] / Ng[g] - mean_remaining ** 2
        
        # Create FastCovarianceStructure
        result[g] = FastCovarianceStructure(
            top_k_cov=cov_top_k,
            remaining_var=var_remaining,
            top_k_indices=top_k_indices,
            remaining_indices=remaining_indices,
            gene_total_expression=gene_total_expression
        )

    if len(groups) == 1:
        return list(result.values())[0] 
    return result


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



###############################################################################
## Helpers to prepare and postprocess copula parameters
###############################################################################


def group_indices(grouping_var: str, obs: pd.DataFrame) -> dict:
    """
    Returns a dictionary of group indices for each group in the grouping variable.
    """
    if grouping_var is None:
        grouping_var = "_copula_group"
        if "copula_group" not in obs.columns:
            obs["_copula_group"] = pd.Categorical(["shared_group"] * len(obs))
    result = {}

    for group in list(obs[grouping_var].dtype.categories):
        result[group] = np.where(obs[grouping_var].values == group)[0]
    return result


def clip(u: np.array, min: float = 1e-5, max: float = 1 - 1e-5) -> np.array:
    u[u < min] = min
    u[u > max] = max
    return u


def format_copula_parameters(parameters: dict, var_names: list):
    '''
    Format the copula parameters into a dictionary of covariance matrices in pandas dataframe format.
    If the covariance is a FastCovarianceStructure, return it as is.
    If the covariance is a dictionary of FastCovarianceStructure objects, return it as is.
    Otherwise, return a dictionary of covariance matrices in pandas dataframe format.
    '''
    covariance = parameters["covariance"]
    
    # Handle FastCovarianceStructure - keep it as is since it has efficient methods
    if isinstance(covariance, FastCovarianceStructure):
        return covariance
    elif isinstance(covariance, dict) and any(isinstance(v, FastCovarianceStructure) for v in covariance.values()):
        # If it's a dict containing FastCovarianceStructure objects, keep as is
        return covariance
    elif type(covariance) is not dict:
        covariance = pd.DataFrame(
            parameters["covariance"], columns=list(var_names), index=list(var_names)
        )
    else:
        for group in covariance.keys():
            if not isinstance(covariance[group], FastCovarianceStructure):
                covariance[group] = pd.DataFrame(
                    parameters["covariance"][group],
                    columns=list(var_names),
                    index=list(var_names),
                )
    return covariance


def strip_dataloader(dataloader, pop=False):
    return DataLoader(
        dataset=dataloader.dataset,
        batch_sampler=dataloader.batch_sampler,
        collate_fn=stack_collate(pop=pop, groups=False),
    )
    