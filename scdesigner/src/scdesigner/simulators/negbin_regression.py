from ..estimators.negbin import negbin_regression
from ..predictors.negbin import negbin_predict
from ..samplers.negbin import negbin_sample
from anndata import AnnData
import pandas as pd


class NegBinRegressionSimulator:
    def __init__(self, **kwargs):
        self.var_names = None
        self.formula = None
        self.params = None
        self.hyperparams = kwargs

    def fit(self, adata: AnnData, formula: str) -> dict:
        self.formula = formula
        self.params = negbin_regression(adata, formula, **self.hyperparams)

    def sample(self, obs: pd.DataFrame) -> AnnData:
        local_parameters = self.predict(obs)
        return negbin_sample(local_parameters, obs)

    def predict(self, obs: pd.DataFrame) -> dict:
        return negbin_predict(self.params, obs, self.formula)

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'Negtive Binomial Regression'
    formula: '{self.formula}'
    parameters: 'beta', 'gamma'"""