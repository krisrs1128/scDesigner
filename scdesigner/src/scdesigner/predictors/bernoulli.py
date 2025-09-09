import numpy as np
import pandas as pd
from ..format import format_matrix


def bernoulli_predict(parameters: dict, obs: pd.DataFrame, formula: str):
    x = format_matrix(obs, formula)
    theta = np.exp(x @ parameters["coef_mean"])
    return {"mean": theta}
