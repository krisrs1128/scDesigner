from .. import data
from anndata import AnnData
from formulaic import model_matrix
from scipy.stats import norm, multivariate_normal
from typing import Union
import numpy as np
import pandas as pd
import torch, scipy


def marginal_aic_bic(likelihood, params: dict, adata: AnnData, 
                     formula: Union[str, dict], allowed_keys: set,
                     param_order: list = None, transform: list = None, 
                     chunk_size: int = int(1e4), batch_size=512):
    device = data.formula.check_device()
    nsample = len(adata)
    params = likelihood_unwrapper(params, param_order, transform).to(device)
    nparam = len(params)

    # create batches for likelihood calculation
    formula = data.standardize_formula(formula, allowed_keys)
    loader = data.multiple_formula_loader(
        adata, formula, chunk_size=chunk_size, batch_size=batch_size
    )
    keys = list(loader.keys())
    loaders = list(loader.values())
    num_keys = len(keys)
    
    ll = 0
    with torch.no_grad():
        for batches in zip(*loaders):
            x = {keys[i]: batches[i][0].to(device) for i in range(num_keys)}
            y = batches[0][1].to(device)
            ll += -likelihood(params, x, y)
    aic = 2 * nparam - 2 * ll
    bic = np.log(nsample) * nparam - 2 * ll
    return aic.cpu().item(), bic.cpu().item()


def gaussian_copula_aic_bic(uniformizer, params: dict, adata: AnnData, 
                            formula: Union[str, dict], allowed_keys: set, copula_groups=None):
    params = uniformizer_unwrapper(params)
    covariance = params['covariance']
    y = adata.X
    if isinstance(y, scipy.sparse._csc.csc_matrix):
        y = y.todense()
    formula = data.standardize_formula(formula, allowed_keys)
    X = {key: model_matrix(formula[key], pd.DataFrame(adata.obs)) for key in formula}
    if copula_groups is not None:
        memberships = adata.obs[copula_groups]
    else:
        copula_groups = "shared_group"
        memberships = np.array(["shared_group"] * y.shape[0])
        
    u = uniformizer(params, X, y)
    groups = covariance.keys()
    nparam = {
        g: (np.sum(covariance[g] != 0) - covariance[g].shape[0]) / 2 for g in groups
    }
    aic = 0 # in the future may add group-wise AIC/BIC
    bic = 0
    for g in groups:
        ix = np.where(memberships == g)[0]
        z = norm().ppf(u[ix])
        copula_ll = multivariate_normal.logpdf(
            z, np.zeros(covariance[g].shape[0]), covariance[g]
        )
        marginal_ll = norm.logpdf(z)
        aic += -2 * (np.sum(copula_ll) - np.sum(marginal_ll)) + 2 * nparam[g]
        bic += (
            -2 * (np.sum(copula_ll) - np.sum(marginal_ll))
            + np.log(z.shape[0]) * nparam[g]
        )
    return aic, bic


def compose_marginal_diagnose(likelihood, allowed_keys: set, param_order: list = None, transform: list = None):
    def diagnose(params: dict, adata: AnnData, formula: Union[str, dict],
                chunk_size: int = int(1e4), batch_size=512):
        return marginal_aic_bic(likelihood, params, adata, formula, allowed_keys,
                                param_order, transform, chunk_size, batch_size)
    return diagnose


def compose_gcopula_diagnose(likelihood, uniformizer, allowed_keys: set, 
                             param_order: list = None, transform: list = None):
    def diagnose(params: dict, adata: AnnData, formula: str, copula_groups=None,
                 chunk_size: int = int(1e4), batch_size=512):
        marginal_aic, marginal_bic = marginal_aic_bic(likelihood, params, adata, formula, allowed_keys,
                                                    param_order, transform, chunk_size, batch_size)
        copula_aic, copula_bic = gaussian_copula_aic_bic(uniformizer, params, adata, formula, allowed_keys, copula_groups)
        return marginal_aic, marginal_bic, copula_aic, copula_bic
    return diagnose


###############################################################################
## Helper for converting params to match likelihood/uniformizer input format
###############################################################################

def likelihood_unwrapper(params: dict, param_order: list = None, transform: list = None):
    l = []
    keys_to_process = [k for k in params if k != "covariance"] if param_order is None else param_order

    for idx, k in enumerate(keys_to_process):
        feature = params[k]
        v = torch.Tensor(feature.values).reshape(1, feature.shape[0] * feature.shape[1])[0]
        if transform is not None:
            v = transform[idx](v)
        l.append(v)

    return torch.cat(l, dim=0)

def uniformizer_unwrapper(params):
    params = params = {key: params[key].values if key!='covariance' else params[key] for key in params}
    if not isinstance(params['covariance'], dict):
        params['covariance'] = {'shared_group': params['covariance'].values}
    else:
        params['covariance'] = {key: params['covariance'][key].values for key in params['covariance']}
    return params
