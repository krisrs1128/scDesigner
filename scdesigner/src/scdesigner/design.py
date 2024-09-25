from formulaic import Formula
import numpy as np
import pandas as pd
import torch


def parse_formula(f, x_names):
    all_names = "+".join(x_names)
    return f.replace("~ .", all_names)


def design(formula, X=None, rows=None):
    if X is None:
        X = pd.DataFrame({"intercept": np.ones((rows))})

    f = parse_formula(formula, X.columns)
    X = Formula(f).get_model_matrix(X)
    cnames = X.columns
    X = np.array(X).astype(np.float32)
    return torch.from_numpy(X), list(cnames)


def reconcile_formulas(formula, terms=["mu", "alpha"]):
    values = formula.values()
    if len(set(values)) == 1:
        return f"""all: {formula[terms[0]]}"""
    return f"""{terms[0]}: {formula[terms[0]]}, {terms[1]}: {formula[terms[1]]}"""


def initialize_formula(f, parameters=["alpha", "mu"], priority="mu"):
    if isinstance(f, str):
        f = {priority: f}

    for k in parameters:
        if k not in f.keys():
            f[k] = "~ 1"
    return f
