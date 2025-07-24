import numpy as np
import pandas as pd
from ..format import format_matrix
from typing import Union

def negbin_predict(parameters: dict, obs: pd.DataFrame, formula: Union[str, dict]):
    # Standardize formula to dictionary format
    if isinstance(formula, str):
        x = format_matrix(obs, formula)
        r, mu = np.exp(parameters["coef_dispersion"]), np.exp(x @ parameters["coef_mean"])
        r = np.repeat(r, mu.shape[0], axis=0)
        return {"mean": mu, "dispersion": r}
    
    x_mean = format_matrix(obs, formula["mean"]) 
    x_dispersion = format_matrix(obs, formula["dispersion"]) # format_matrix returns a pandas dataframe
    
    r = np.exp(x_dispersion @ parameters["coef_dispersion"])
    mu = np.exp(x_mean @ parameters["coef_mean"])
    # r and mu are still dataframes with column names being the gene names
    return {"mean": mu, "dispersion": r}
