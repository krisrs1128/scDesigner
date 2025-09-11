import numpy as np
import pandas as pd
import anndata as ad
from typing import Union
from scipy.stats import norm


def glm_sample_factory(sample_array):
    def sampler(local_parameters: dict, obs: pd.DataFrame) -> ad.AnnData:
        samples = sample_array(local_parameters)
        result = ad.AnnData(X=samples, obs=obs)
        result.var_names = local_parameters["mean"].columns
        return result
    return sampler

def gaussian_copula_pseudo_obs(N, G, sigma, groups):

    # Import here to avoid circular imports
    from ..estimators.gaussian_copula_factory import FastCovarianceStructure
    
    u = np.zeros((N, G))

    # cycle across groups
    for group, ix in groups.items():
        # If sigma is not a dict, then every group shares the same sigma
        if type(sigma) is not dict:
            sigma = {group: sigma}
        
        group_sigma = sigma[group]
    
        # Handle FastCovarianceStructure
        if isinstance(group_sigma, FastCovarianceStructure):
            u[ix] = _fast_copula_pseudo_obs(len(ix), group_sigma)
        else:
            # Traditional full covariance matrix approach
            z = np.random.multivariate_normal(
                mean=np.zeros(G), cov=group_sigma, size=len(ix)
            )
            normal_distn = norm(0, np.diag(group_sigma ** 0.5))
            u[ix] = normal_distn.cdf(z)
    return u


def _fast_copula_pseudo_obs(n_samples, fast_cov_struct):
    """
    Efficient pseudo-observation generation using FastCovarianceStructure.
    
    This function separately samples:
    1. Top-k genes using full multivariate normal with their covariance matrix
    2. Remaining genes using independent normal with their individual variances
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate for this group
    fast_cov_struct : FastCovarianceStructure
        Structure containing top-k covariance and remaining variances
        
    Returns:
    --------
    np.ndarray : Pseudo-observations with shape (n_samples, total_genes)
    """
    u = np.zeros((n_samples, fast_cov_struct.total_genes))
    
    # Sample top-k genes with full covariance
    if fast_cov_struct.top_k > 0:
        z_top_k = np.random.multivariate_normal(
            mean=np.zeros(fast_cov_struct.top_k), 
            cov=fast_cov_struct.top_k_cov, 
            size=n_samples
        )
        
        # Convert to uniform via marginal CDFs
        top_k_std = np.sqrt(np.diag(fast_cov_struct.top_k_cov))
        normal_distn_top_k = norm(0, top_k_std)
        u[:, fast_cov_struct.top_k_indices] = normal_distn_top_k.cdf(z_top_k)
    
    # Sample remaining genes independently
    if len(fast_cov_struct.remaining_indices) > 0:
        remaining_std = np.sqrt(fast_cov_struct.remaining_var)
        z_remaining = np.random.normal(
            loc=0, 
            scale=remaining_std, 
            size=(n_samples, len(fast_cov_struct.remaining_indices))
        )
        
        # Convert to uniform via marginal CDFs  
        normal_distn_remaining = norm(0, remaining_std)
        u[:, fast_cov_struct.remaining_indices] = normal_distn_remaining.cdf(z_remaining)
    
    return u


def gaussian_copula_sample_factory(copula_sample_array):
    def sampler(
        local_parameters: dict, covariance: Union[dict, np.array], groups: dict, obs: pd.DataFrame
    ) -> ad.AnnData:
        samples = copula_sample_array(local_parameters, covariance, groups)
        result = ad.AnnData(X=samples, obs=obs)
        result.var_names = local_parameters["mean"].columns
        return result
    return sampler

