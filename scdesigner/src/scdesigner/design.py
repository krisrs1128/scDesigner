from formulaic import Formula
import numpy as np
import pandas as pd
import torch


def parse_formula(f, x_names):
    all_names = "+".join(x_names)
    return f.replace("~ .", all_names)


def design(formula, X):
    if X is None:
        X = pd.DataFrame({"intercept": np.ones((X.shape[0]))})

    f = parse_formula(formula, X.columns)
    X = Formula(f).get_model_matrix(X)
    cnames = X.columns
    X = np.array(X).astype(np.float32)
    return torch.from_numpy(X), list(cnames)
