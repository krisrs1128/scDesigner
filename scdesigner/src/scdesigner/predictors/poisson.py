import numpy as np
import pandas as pd
from ..format import format_matrix
from typing import Union

def poisson_predict(parameters: dict, obs: pd.DataFrame, formula: Union[str, dict]):
    if isinstance(formula, dict):
        formula = formula['mean']
    x = format_matrix(obs, formula)
    mu = np.exp(x @ parameters["coef_mean"])
    return {"mean": mu}

