from ..data import formula_group_loader
from anndata import AnnData
from collections.abc import Callable
from copy import deepcopy
from formulaic import model_matrix
from scipy.stats import norm
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch

###############################################################################
## General copula factory functions
###############################################################################


def remove_group_collate(batch):
    x = torch.stack([x[0] for x in batch])
    y = torch.stack([x[1] for x in batch])
    return [x, y]


def gaussian_copula_array_factory(marginal_model: Callable, uniformizer: Callable):
    def copula_fun(loader: DataLoader, **kwargs):
        # for the marginal model, ignore the groupings
        formula_loader = deepcopy(loader)
        formula_loader.collate_fn = remove_group_collate(formula_loader)
        parameters = marginal_model(formula_loader, **kwargs)

        # estimate covariance, allowing for different groups
        parameters["covariance"] = copula_covariance(parameters, loader, uniformizer)
        return parameters

    return copula_fun


def gaussian_copula_factory(copula_array_fun: Callable, parameter_formatter: Callable):
    def copula_fun(
        adata: AnnData, formula: str = "~ 1", grouping_variable: str = None, **kwargs
    ) -> dict:
        dl = formula_group_loader(adata, formula, grouping_variable)
        parameters = copula_array_fun(dl, **kwargs)
        parameters = parameter_formatter(
            parameters, adata.var_names, dl.dataset.x_names
        )
        parameters["covariance"] = format_copula_parameters(parameters, adata.var_names)
        return parameters

    return copula_fun


def copula_covariance(parameters: dict, loader: DataLoader, uniformizer: Callable):
    D = u.shape[1]
    result = {g: np.zeros((D, D)) for g in loader.dataset.groups}

    for memberships, x, y in loader:
        u = uniformizer(parameters, x, y)
        for g in result.keys():
            ix = np.where(memberships == g)
            z = norm().ppf(u[ix]).T
            result[g] += z @ z.T

    if len(result) == 1:
        return list(result.values())[0]
    return result


###############################################################################
## Helpers to prepare and postprocess copula parameters
###############################################################################


def group_indices(formula: str, obs: pd.DataFrame) -> dict:
    group_matrix = model_matrix(formula, obs)
    result = {}

    for group in group_matrix.columns:
        result[group] = np.where(group_matrix[group].values == 1)[0]
    return result


def clip(u: np.array, min: float = 1e-5, max: float = 1 - 1e-5) -> np.array:
    u[u < min] = min
    u[u > max] = max
    return u


def format_copula_parameters(parameters: dict, var_names: list):
    covariance = parameters["covariance"]
    if type(covariance) is not dict:
        covariance = pd.DataFrame(
            parameters["covariance"], columns=list(var_names), index=list(var_names)
        )
    else:
        for group in covariance.keys():
            covariance[group] = pd.DataFrame(
                parameters["covariance"][group],
                columns=list(var_names),
                index=list(var_names),
            )
    return covariance
