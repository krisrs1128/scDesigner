from ..format import format_matrix
import numpy as np
import pandas as pd


def zero_inflated_poisson_predict(parameters: dict, obs: pd.DataFrame, formula: str):
    x = format_matrix(obs, formula)
    mu, pi = (
        np.exp(x @ parameters["beta"]),
        parameters["pi"],
    )
    pi = np.repeat(pi, mu.shape[0], axis=0)
    return {"mean": mu, "zero_inflation": pi}
