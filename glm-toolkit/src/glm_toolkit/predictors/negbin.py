import numpy as np
import pandas as pd
from ..format import format_matrix


def negbin_predict(parameters: dict, obs: pd.DataFrame, formula: str):
    x = format_matrix(obs, formula)
    r, mu = np.exp(parameters["gamma"]), np.exp(x @ parameters["beta"])
    r = np.repeat(r, mu.shape[0], axis=0)
    return {"mean": mu, "dispersion": r}
