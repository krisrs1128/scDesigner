from .glm_regression import format_input_anndata
from anndata import AnnData
from collections.abc import Callable
from formulaic import model_matrix
from scipy.stats import norm
import numpy as np
import pandas as pd

###############################################################################
## General copula factory functions
###############################################################################


def gaussian_copula_array_factory(marginal_model: Callable, uniformizer: Callable):
    def copula_fun(x: np.array, y: np.array, groups: dict, **kwargs):
        parameters = marginal_model(x, y, **kwargs)
        u = uniformizer(parameters, x, y)
        parameters["covariance"] = copula_covariance(u, groups)
        return parameters

    return copula_fun


def gaussian_copula_factory(copula_array_fun: Callable, parameter_formatter: Callable):
    def copula_fun(
        adata: AnnData, formula: str = "~ 1", formula_copula: str = "~ 1", **kwargs
    ) -> dict:
        adata = format_input_anndata(adata)
        x = model_matrix(formula, adata.obs)

        groups = group_indices(formula_copula, adata.obs)
        parameters = copula_array_fun(np.array(x), adata.X, groups, **kwargs)
        parameters = parameter_formatter(parameters, adata.var_names, x.columns)
        parameters["covariance"] = format_copula_parameters(parameters, adata.var_names)
        return parameters

    return copula_fun


def copula_covariance(u: np.array, groups: dict):
    result = {}
    for group, ix in groups.items():
        result[group] = np.cov(norm().ppf(u[ix]).T)

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
