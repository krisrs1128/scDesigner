from anndata import AnnData
from formulaic import model_matrix
from scipy.stats import nbinom
import scipy.sparse
from ..estimators.glm_regression import (
    negative_binomial_regression_array,
    format_input_anndata,
    format_nb_parameters,
)
import numpy as np
import pandas as pd


class NegBinRegressionSimulator:
    def __init__(self):  # default input: cell x gene
        self.var_names = None
        self.formula = None
        self.shape = None

    def estimate(self, adata: AnnData, formula: str, **kwargs) -> dict:
        adata = format_input_anndata(adata)
        self.formula = formula
        self.shape = adata.X.shape
        x = model_matrix(formula, adata.obs)
        parameters = negative_binomial_regression_array(np.array(x), adata.X, **kwargs)
        return format_nb_parameters(parameters, list(adata.var_names), list(x.columns))

    def sample(self, parameters: dict, obs: pd.DataFrame, formula=None) -> AnnData:
        if formula is not None:
            x = model_matrix(formula, obs)
        else:
            x = obs

        params = self.predict(parameters, x)
        r, mu = params["dispersion"], params["coefficient"]
        samples = nbinom(n=r, p=r / (r + mu)).rvs()
        result = AnnData(X=samples, obs=obs)
        result.var_names = parameters["dispersion"].columns
        return result

    def predict(self, parameters: dict, obs: pd.DataFrame, formula=None) -> dict:
        if formula is not None:
            x = model_matrix(formula, obs)
        else:
            x = obs

        r, mu = np.exp(parameters["dispersion"]), np.exp(x @ parameters["coefficient"])
        r = np.repeat(r, mu.shape[0], axis=0)
        return {"coefficient": mu, "dispersion": r}

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'Negtive Binomial Regression'
    formula: '{self.formula}'
    parameters: 'coefficient', 'dispersion'"""
