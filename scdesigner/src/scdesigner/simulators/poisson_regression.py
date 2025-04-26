from ..estimators.poisson import poisson_regression
from ..predictors.poisson import poisson_predict
from ..samplers.poisson import poisson_sample
from anndata import AnnData
import pandas as pd


class PoissonRegressionSimulator:
    def __init__(self, **kwargs):
        self.formula = None
        self.params = None
        self.hyperparams = kwargs

    def fit(self, adata: AnnData, formula: str) -> dict:
        self.formula = formula
        self.params = poisson_regression(adata, formula, **self.hyperparams)

    def sample(self, obs: pd.DataFrame) -> AnnData:
        local_parameters = self.predict(obs)
        return poisson_sample(local_parameters, obs)

    def predict(self, obs: pd.DataFrame) -> dict:
        return poisson_predict(self.params, obs, self.formula)

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'Poisson Regression'
    formula: '{self.formula}'
    parameters: 'beta'"""