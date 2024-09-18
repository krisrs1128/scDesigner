from formulaic import Formula
import numpy as np
import pandas as pd
import torch


def parse_formula(f, x_names):
    all_names = "+".join(x_names)
    return f.replace("~ .", all_names)


def design(formula, X, Y):
    if X is None:
        X = pd.DataFrame({"intercept": np.ones((Y.shape[0]))})

    f = parse_formula(formula, X.columns)
    X = Formula(f).get_model_matrix(X, output="numpy")
    X = np.array(X).astype(np.float32)
    return torch.from_numpy(X)
