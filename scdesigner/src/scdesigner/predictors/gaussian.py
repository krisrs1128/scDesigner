import numpy as np
import pandas as pd
from ..format import format_matrix
from typing import Union

def gaussian_predict(parameters: dict, obs: pd.DataFrame, formula: Union[str, dict]):
    # Standardize formula to dictionary format
    if isinstance(formula, str):
        formula = {'mean': formula, 'sdev': '~ 1'}
    
    x_mean = format_matrix(obs, formula["mean"]) 
    x_dispersion = format_matrix(obs, formula["sdev"])
    
    sigma = np.exp(x_dispersion @ parameters["coef_sdev"])
    mu = x_mean @ parameters["coef_mean"]
    return {"mean": mu, "sdev": sigma}
