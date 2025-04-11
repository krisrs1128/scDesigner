from anndata import AnnData
from ..format.format import format_input_anndata
from ..estimators.negbin import negbin_copula
from ..predictors.negbin import negbin_predict
from ..estimators.gaussian_copula_factory import group_indices
from ..samplers.negbin import negbin_copula_sample
import pandas as pd


class NegBinCopulaSimulator:
    def __init__(self, **kwargs):
        self.var_names = None
        self.formula = None
        self.copula_formula = None
        self.shape = None
        self.params = None
        self.hyperparams = kwargs

    def fit(
        self,
        adata: AnnData,
        formula: str = "~ 1",
        formula_copula: str = "~ 1"
    ) -> dict:

        adata = format_input_anndata(adata)
        self.formula = formula
        self.copula_formula = formula_copula
        self.shape = adata.X.shape
        self.params = negbin_copula(adata, formula, formula_copula, self.hyperparams)

    def sample(self, obs: pd.DataFrame) -> AnnData:
        groups = group_indices(self.copula_formula, obs)
        local_parameters = self.predict(obs)
        return negbin_copula_sample(
            local_parameters, self.params["covariance"], groups, obs
        )

    def predict(self, obs: pd.DataFrame) -> dict:
        return negbin_predict(self.params, obs, self.formula)

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'Negtive Binomial Copula'
    formula: '{self.formula}'
    copula formula: '{self.copula_formula}'
    parameters: 'coefficient', 'dispersion', 'covariance'"""