from ..format import format_matrix
from typing import Union
import numpy as np
import pandas as pd


def zero_inflated_poisson_predict(parameters: dict, obs: pd.DataFrame, formula: Union[str, dict]):
    if isinstance(formula, str):
        formula = {'beta': formula, 'pi': '~ 1'}
    mu, pi = (
        np.exp(format_matrix(obs, formula['mean']) @ parameters["coef_mean"]),
        sigmoid(format_matrix(obs, formula['zero_inflation']) @ parameters["coef_zero_inflation"]),
    )
    return {"mean": mu, "zero_inflation": pi}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))