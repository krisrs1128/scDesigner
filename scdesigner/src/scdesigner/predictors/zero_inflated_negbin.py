import numpy as np
import pandas as pd
from ..format import format_matrix


def zero_inflated_negbin_predict(parameters: dict, obs: pd.DataFrame, formula: str):
    x = format_matrix(obs, formula)
    r, mu, pi = (
        np.exp(parameters["gamma"]),
        np.exp(x @ parameters["beta"]),
        parameters["pi"],
    )
    r = np.repeat(r, mu.shape[0], axis=0)

    return {"mean": mu, "dispersion": r, "zero_inflation": pi}
