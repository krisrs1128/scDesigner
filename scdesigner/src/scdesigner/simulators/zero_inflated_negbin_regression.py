from ..estimators.zero_inflated_negbin import zero_inflated_negbin_regression
from ..predictors.zero_inflated_negbin import zero_inflated_negbin_predict
from ..samplers.zero_inflated_negbin import zero_inflated_negbin_sample
from anndata import AnnData
import pandas as pd


class ZeroInflatedNegbinRegressionSimulator:
    def __init__(self, **kwargs):
        self.formula = None
        self.params = None
        self.hyperparams = kwargs

    def fit(self, adata: AnnData, formula: str) -> dict:
        self.formula = formula
        self.params = zero_inflated_negbin_regression(adata, formula, **self.hyperparams)

    def sample(self, obs: pd.DataFrame) -> AnnData:
        local_parameters = self.predict(obs)
        return zero_inflated_negbin_sample(local_parameters, obs)

    def predict(self, obs: pd.DataFrame) -> dict:
        return zero_inflated_negbin_predict(self.params, obs, self.formula)

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'Zero-Inflated Negtive Binomial Regression'
    formula: '{self.formula}'
    parameters: 'beta', 'gamma'"""