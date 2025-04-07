import numpy as np
import pandas as pd
from ..estimators.format import format_matrix


def bernoulli_predict(parameters: dict, obs: pd.DataFrame, formula: str):
    x = format_matrix(obs, formula)
    theta = np.exp(x @ parameters["beta"])
    return {"mean": theta}
