from ..estimators.bernoulli import bernoulli_regression
from ..predictors.bernoulli import bernoulli_predict
from ..samplers.bernoulli import bernoulli_sample
from anndata import AnnData
import pandas as pd


class BernoulliRegressionSimulator:
    def __init__(self, **kwargs):
        self.formula = None
        self.params = None
        self.hyperparams = kwargs

    def fit(self, adata: AnnData, formula: str) -> dict:
        self.formula = formula
        self.params = bernoulli_regression(adata, formula, **self.hyperparams)

    def sample(self, obs: pd.DataFrame) -> AnnData:
        local_parameters = self.predict(obs)
        return bernoulli_sample(local_parameters, obs)

    def predict(self, obs: pd.DataFrame) -> dict:
        return bernoulli_predict(self.params, obs, self.formula)

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'Bernoulli Regression'
    formula: '{self.formula}'
    parameters: 'beta'"""