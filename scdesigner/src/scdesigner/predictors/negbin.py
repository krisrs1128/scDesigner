import numpy as np
import pandas as pd
from ..format import format_matrix
from typing import Union

def negbin_predict(parameters: dict, obs: pd.DataFrame, formula: Union[str, dict]):
    # Standardize formula to dictionary format
    if isinstance(formula, str):
        formula = {'mean': formula, 'dispersion': '~ 1'}
    
    x_mean = format_matrix(obs, formula["mean"]) 
    x_dispersion = format_matrix(obs, formula["dispersion"]) # format_matrix returns a pandas dataframe
    
    # Update parameter keys to match the current naming convention
    r = np.exp(x_dispersion @ parameters["beta_dispersion"])
    mu = np.exp(x_mean @ parameters["beta_mean"])
    # r and mu are still dataframes with column names being the gene names
    return {"mean": mu, "dispersion": r}
