from ..data import formula_group_loader, stack_collate
from anndata import AnnData
from collections.abc import Callable
from formulaic import model_matrix
from scipy.stats import norm
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

###############################################################################
## General copula factory functions
###############################################################################


def gaussian_copula_array_factory(marginal_model: Callable, uniformizer: Callable):
    def copula_fun(loader: DataLoader, lr: float = 0.1, epochs: int = 40, **kwargs):
        # for the marginal model, ignore the groupings
        formula_loader = strip_dataloader(loader, pop="Stack" in type(loader.dataset).__name__)
        parameters = marginal_model(formula_loader, lr=lr, epochs=epochs, **kwargs)

        # estimate covariance, allowing for different groups
        parameters["covariance"] = copula_covariance(parameters, loader, uniformizer)
        return parameters

    return copula_fun


def gaussian_copula_factory(copula_array_fun: Callable, parameter_formatter: Callable):
    def copula_fun(
        adata: AnnData,
        formula: str = "~ 1",
        grouping_variable: str = None,
        chunk_size: int = int(1e4),
        batch_size: int = 512,
        **kwargs
    ) -> dict:
        dl = formula_group_loader(
            adata,
            formula,
            grouping_variable,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )
        parameters = copula_array_fun(dl, **kwargs)
        parameters = parameter_formatter(
            parameters, adata.var_names, dl.dataset.x_names
        )
        parameters["covariance"] = format_copula_parameters(parameters, adata.var_names)
        return parameters

    return copula_fun


def copula_covariance(parameters: dict, loader: DataLoader, uniformizer: Callable):
    D = next(iter(loader))[1].shape[1]
    groups = loader.dataset.groups
    sums = {g: np.zeros(D) for g in groups}
    second_moments = {g: np.eye(D) for g in groups}
    Ng = {g: 0 for g in groups}

    for x, y, memberships in loader:
        u = uniformizer(parameters, x.cpu().numpy(), y.cpu().numpy())
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


def strip_dataloader(dataloader, pop=False):
    return DataLoader(
        dataset=dataloader.dataset,
        batch_sampler=dataloader.batch_sampler,
        collate_fn=stack_collate(pop=pop, groups=False),
    )