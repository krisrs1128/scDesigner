from typing import Union, Sequence
import numpy as np
import pandas as pd
import re
import torch
import copy
from ..base.copula import CovarianceStructure


def nullify(sim, row_pattern: str, col_pattern: str, param: str):
    """Nullify marginal parameters

    Zero out entries matching (row_pattern, col_pattern) for the marginal `param`.
    """
    sim = copy.deepcopy(sim)
    df = sim.parameters["marginal"][param]
    matches = data_frame_mask(df, row_pattern, col_pattern)
    mask = (~matches).astype(float)
    mat = sim.marginal.predict.coefs[param].detach().cpu().numpy() * mask
    _update_marginal_param(sim, param, mat)
    return sim


def amplify(sim, factor: float, row_pattern: str, col_pattern: str, param: str):
    """Multiply selected marginal entries by factor."""
    sim = copy.deepcopy(sim)
    df = sim.parameters["marginal"][param]
    matches = data_frame_mask(df, row_pattern, col_pattern).astype(float)
    mask = factor * matches + np.ones_like(matches)
    mat = sim.marginal.predict.coefs[param].detach().cpu().numpy() * mask
    _update_marginal_param(sim, param, mat)
    return sim


def decorrelate(sim, row_pattern: str, col_pattern: str, group: Union[str, None] = None):
    """Zero out selected off-diagonal entries of a covariance.
    """
    decorr_sim = copy.deepcopy(sim)
    decorr_sim.copula.decorrelate(row_pattern, col_pattern, group)
    return decorr_sim


def correlate(sim, factor: float, row_pattern: str, col_pattern: str, group: Union[str, None] = None):
    """Multiply selected off-diagonal entries by factor."""
    corr_sim = copy.deepcopy(sim)
    corr_sim.copula.correlate(factor, row_pattern, col_pattern, group)
    return corr_sim


def replace_param(sim, path: Sequence[str], new_param: Union[np.ndarray, pd.DataFrame, CovarianceStructure]):
    """Substitute a new parameter for an old one.

    Use the path to the parameter starting from sim.parameters to identify the
    parameter to transform.  Examples: ['marginal','mean'] or
    ['copula','group_name']
    
    Args:
        sim (Simulator): The simulator object.
        path (Sequence[str]): The path to the parameter to transform.
        new_param (np.ndarray): The new parameter to substitute. 
        For replacing a covariance structure, new_param could be a numpy array of shape (n_genes, n_genes)
        or a CovarianceStructure object defined by the user.
    """
    sim = copy.deepcopy(sim)
    if path[0] == "marginal":
        param = path[1]
        mat = np.asarray(new_param)
        _update_marginal_param(sim, param, mat)

    if path[0] == "copula":
        if isinstance(new_param, np.ndarray):
            sim.parameters["copula"][path[1]] = CovarianceStructure(new_param,
                                                                    modeled_names=sim.adata.var_names)
        elif isinstance(new_param, pd.DataFrame):
            sim.parameters["copula"][path[1]] = CovarianceStructure(new_param.values,
                                                                    modeled_names=new_param.index)
        elif isinstance(new_param, CovarianceStructure):
            sim.parameters["copula"][path[1]] = new_param
        else:
            raise ValueError(f"new_param must be a numpy array or a CovarianceStructure object, got {type(new_param)}")
    return sim


###############################################################################
## Helper functions used throughout
###############################################################################

def str_match(string: str, string_list: list) -> bool:
    for l in string_list:
        if l in string:
            return True
    return False


def data_frame_mask(df, row_pattern=".", col_pattern=".") -> np.array:
    """Return a boolean mask for a pandas DataFrame where rows/cols match regex patterns.

    The returned mask has shape (n_rows, n_cols) and is True where the
    intersection of matched rows and matched columns occurs.
    """
    mask = np.zeros(df.shape, dtype=bool)
    if isinstance(col_pattern, str):
        col_pattern = [col_pattern]
    if isinstance(row_pattern, str):
        row_pattern = [row_pattern]

    # check for columns that match at least one pattern
    col_matches = np.zeros(df.shape[1], dtype=bool)
    for col in df.columns:
        if any(re.search(pat, col) for pat in col_pattern):
            col_matches[df.columns.get_loc(col)] = True

    # same with rows
    row_matches = np.zeros(df.shape[0], dtype=bool)
    for idx in df.index:
        if any(re.search(pat, str(idx)) for pat in row_pattern):
            row_matches[df.index.get_loc(idx)] = True

    # set mask to the intersection of row and column matches
    mask = np.outer(row_matches, col_matches)
    return mask


def _apply_to_groups(cov_obj, grp, f):
    """Apply f to either all group arrays or a single group's array."""
    if isinstance(cov_obj, dict):
        if grp is None:
            for k in list(cov_obj.keys()):
                f(cov_obj[k])
        else:
            f(cov_obj[grp])
    else:
        f(cov_obj)


def _update_marginal_param(sim, param: str, mat: np.ndarray):
    """Update the torch Parameter for a marginal `param`"""
    tensor = sim.marginal.predict.coefs[param]
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(mat)).to(dtype=tensor.dtype, device=tensor.device)
        tensor.copy_(t)
    sim.parameters["marginal"][param].values[:] = tensor.detach().cpu().numpy()