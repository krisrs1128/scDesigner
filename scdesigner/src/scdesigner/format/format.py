from anndata import AnnData
from formulaic import model_matrix
import pandas as pd
import scipy.sparse

def to_np(x):
    return x.detach().cpu().numpy()

def format_input_anndata(adata: AnnData) -> AnnData:
    result = adata.copy()
    if isinstance(result.X, scipy.sparse._csc.csc_matrix):
        result.X = result.X.todense()
    return result

def format_matrix(obs: pd.DataFrame, formula: str):
    if formula is not None:
        x = model_matrix(formula, pd.DataFrame(obs))
    else:
        x = obs
    return x