from ..estimators.negbin import negbin_regression
from ..predictors.negbin import negbin_predict
from ..samplers.negbin import negbin_sample
from anndata import AnnData
import pandas as pd
import scipy.sparse


class NegBinRegressionSimulator:
    def __init__(self):
        self.var_names = None
        self.formula = None
        self.shape = None

    def estimate(self, adata: AnnData, formula: str, **kwargs) -> dict:
        self.formula = formula
        return negbin_regression(adata, formula, **kwargs)

    def sample(self, parameters: dict, obs: pd.DataFrame) -> AnnData:
        local_parameters = self.predict(parameters, obs)
        return negbin_sample(local_parameters, obs)

    def predict(self, parameters: dict, obs: pd.DataFrame) -> dict:
        return negbin_predict(parameters, obs, self.formula)

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'Negtive Binomial Regression'
    formula: '{self.formula}'
    parameters: 'beta', 'gamma'"""


###############################################################################
## Helpers for processing input data
###############################################################################


def format_input_anndata(adata: AnnData) -> AnnData:
    result = adata.copy()
    if isinstance(result.X, scipy.sparse._csc.csc_matrix):
        result.X = result.X.todense()
    return result
