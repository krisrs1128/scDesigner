from anndata import AnnData
from formulaic import model_matrix
from ..format import format_input_anndata, format_matrix
from scipy.stats import gamma
from ..estimators.pnmf import pnmf, gamma_regression_array, format_gamma_parameters
import numpy as np
import pandas as pd


class PNMFRegressionSimulator:
    def __init__(self, nbase=20, maxIter=100, **kwargs):  # default input: cell x gene
        self.var_names = None
        self.formula = None
        self.params = None
        self.hyperparams = {"pnmf": {"nbase": nbase, "maxIter": maxIter}, "gamma": kwargs}

    def fit(self, adata, formula: str):
        adata = format_input_anndata(adata)
        self.var_names = adata.var_names
        self.formula = formula
        log_data = np.log1p(adata.X).T
        W, S = pnmf(log_data, **self.hyperparams["pnmf"])
        adata = AnnData(X=S.T, obs=adata.obs)

        x = model_matrix(formula, adata.obs)
        parameters = gamma_regression_array(np.array(x), adata.X, **self.hyperparams["gamma"])
        parameters["W"] = W
        self.params = format_gamma_parameters(
            parameters, list(self.var_names), list(x.columns)
        )

    def sample(self, obs: pd.DataFrame) -> AnnData:
        W = self.params["W"]
        params = self.predict(obs)
        a, loc, beta = params["a"], params["loc"], params["beta"]
        sim_score = gamma(a, loc, 1 / beta).rvs()
        samples = np.exp(W @ sim_score.T).T

        # thresholding samples
        floor = np.floor(samples)
        samples = floor + np.where(samples - floor < 0.9, 0, 1) - 1
        samples = np.where(samples < 0, 0, samples)

        result = AnnData(X=samples, obs=obs)
        result.var_names = self.var_names
        return result

    def predict(self, obs: pd.DataFrame) -> dict:
        x = format_matrix(obs, self.formula)
        a, loc, beta = (
            x @ np.exp(self.params["a"]),
            x @ self.params["loc"],
            x @ np.exp(self.params["beta"]),
        )
        return {"a": a, "loc": loc, "beta": beta}

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'PNMF Regression'
    formula: '{self.formula}'
    parameters: 'a', 'loc', 'beta', 'W'"""
