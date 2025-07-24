import numpy as np
import pandas as pd
from ..format import format_matrix
from scipy.special import expit
from typing import Union

def zero_inflated_negbin_predict(parameters: dict, obs: pd.DataFrame, formula: Union[str, dict]):
    if isinstance(formula, str):
        formula = {"mean": formula, "dispersion": "~ 1", "zero_inflation": "~ 1"}
    x_mean = format_matrix(obs, formula["mean"])
    x_dispersion = format_matrix(obs, formula["dispersion"])
    x_zero_inflation = format_matrix(obs, formula["zero_inflation"])
    r, mu, pi = (
        np.exp(x_dispersion @ parameters["coef_dispersion"]),
        np.exp(x_mean @ parameters["coef_mean"]),
        expit(x_zero_inflation @ parameters["coef_zero_inflation"]),
    )
    return {"mean": mu, "dispersion": r, "zero_inflation": pi}
