import numpy as np
import pandas as pd
from ..format import format_matrix
from scipy.special import expit


def zero_inflated_negbin_predict(parameters: dict, obs: pd.DataFrame, formula: dict):
    x_mean = format_matrix(obs, formula["mean"])
    x_dispersion = format_matrix(obs, formula["dispersion"])
    x_zero_inflation = format_matrix(obs, formula["zero_inflation"])
    r, mu, pi = (
        np.exp(x_dispersion @ parameters["beta_dispersion"]),
        np.exp(x_mean @ parameters["beta_mean"]),
        expit(x_zero_inflation @ parameters["beta_zero_inflation"]),
    )
    return {"mean": mu, "dispersion": r, "zero_inflation": pi}
