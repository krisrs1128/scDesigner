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
    