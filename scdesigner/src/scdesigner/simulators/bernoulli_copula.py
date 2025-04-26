from anndata import AnnData
from ..estimators.bernoulli import bernoulli_copula
from ..predictors.bernoulli import bernoulli_predict
from ..estimators.gaussian_copula_factory import group_indices
from ..samplers.bernoulli import bernoulli_copula_sample
import pandas as pd


class BernoulliCopulaSimulator:
    def __init__(self, **kwargs):
        self.formula = None
        self.copula_groups = None
        self.params = None
        self.hyperparams = kwargs

    def fit(
        self,
        adata: AnnData,
        formula: str = "~ 1",
        copula_groups: str = None
    ) -> dict:
        self.formula = formula
        self.coupla_groups = copula_groups
        self.params = bernoulli_copula(adata, formula, copula_groups, **self.hyperparams)

    def sample(self, obs: pd.DataFrame) -> AnnData:
        groups = group_indices(self.coupla_groups, obs)
        local_parameters = self.predict(obs)
        return bernoulli_copula_sample(
            local_parameters, self.params["covariance"], groups, obs
        )

    def predict(self, obs: pd.DataFrame) -> dict:
        return bernoulli_predict(self.params, obs, self.formula)

    def __repr__(self):
        return f"""scDesigner simulator object with
    method: 'Bernoulli Copula'
    formula: '{self.formula}'
    copula formula: '{self.copula_groups}'
    parameters: 'beta', 'covariance'"""