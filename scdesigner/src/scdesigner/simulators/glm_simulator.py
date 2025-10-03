from .. import estimators as est
from .. import predictors as prd
from .. import samplers as smp
from .. import diagnose
from anndata import AnnData
import pandas as pd
from typing import Union


def glm_simulator_generator(class_name, regressor, sampler, predictor, diagnose):
    def __init__(self, **kwargs):
        self.formula = None # formula should be a string or a dictionary of strings
        self.params = None
        self.marginal_aic = None
        self.marginal_bic = None
        self.copula_aic = None
        self.copula_bic = None
        self.hyperparams = kwargs
        self.filtered_kwargs = {k: kwargs[k] for k in ["chunk_size", "batch_size"] if k in kwargs}

    if "Copula" not in class_name:
        # fitting and sampling methods for plain regressors
        def fit(self, adata: AnnData, formula: Union[str, dict]) -> dict:
            self.formula = formula
            self.params = regressor(adata, formula, **self.hyperparams)
            self.marginal_aic, self.marginal_bic = diagnose(
                self.params, adata, formula, **self.filtered_kwargs
            )

        def sample(self, obs: pd.DataFrame) -> AnnData:
            local_parameters = self.predict(obs) # a dictionary of parameters
            return sampler(local_parameters, obs)

    else:
        # fitting and sampling for gaussian copula models

        def fit(
            self, adata: AnnData, formula: Union[str, dict] = "~ 1", copula_groups: str = None
        ) -> dict:
            self.formula = formula
            self.copula_groups = copula_groups
            self.params = regressor(adata, formula, copula_groups, **self.hyperparams)
            self.marginal_aic, self.marginal_bic, self.copula_aic, self.copula_bic = diagnose(
                self.params, adata, formula, copula_groups, **self.filtered_kwargs
            )

        def sample(self, obs: pd.DataFrame) -> AnnData:
            groups = est.gaussian_copula_factory.group_indices(self.copula_groups, obs)
            local_parameters = self.predict(obs)
            return sampler(local_parameters, self.params["covariance"], groups, obs)

    def predict(self, obs: pd.DataFrame) -> dict:
        return predictor(self.params, obs, self.formula)
        # The predictor function should handle different formula types: dict or string

    def __repr__(self):
        params_string = ", ".join(
            [
                f"{k} [{self.params[k].shape[0]}x{self.params[k].shape[1]}]"
                for k in self.params.keys()
            ]
        )
        return f"""scDesigner simulator object with
    method: {self.__class__.__name__}
    formula: {self.formula}
    parameters: {params_string}"""

    return type(
        class_name,
        (),
        {
            "__init__": __init__,
            "fit": fit,
            "sample": sample,
            "predict": predict,
            "__repr__": __repr__,
        },
    )


NegBinRegressionSimulator = glm_simulator_generator(
    "NegBinRegressionSimulator",
    est.negbin_regression,
    smp.negbin_sample,
    prd.negbin_predict,
    diagnose.negbin_regression_diagnose
)

NegBinCopulaSimulator = glm_simulator_generator(
    "NegBinCopulaSimulator",
    est.negbin_copula, 
    smp.negbin_copula_sample,
    prd.negbin_predict,
    diagnose.negbin_gcopula_diagnose,
)

PoissonRegressionSimulator = glm_simulator_generator(
    "PoissonRegressionSimulator",
    est.poisson_regression,
    smp.poisson_sample,
    prd.poisson_predict,
    diagnose.poisson_regression_diagnose,
)

PoissonCopulaSimulator = glm_simulator_generator(
    "PoissonCopulaSimulator",
    est.poisson_copula,
    smp.poisson_copula_sample,
    prd.poisson_predict,
    diagnose.poisson_gcopula_diagnose,
)

BernoulliRegressionSimulator = glm_simulator_generator(
    "BernoulliRegressionSimulator",
    est.bernoulli_regression,
    smp.bernoulli_sample,
    prd.bernoulli_predict,
    diagnose.bernoulli_regression_diagnose,
)
 
BernoulliCopulaSimulator = glm_simulator_generator(
    "BernoulliCopulaSimulator",
    est.bernoulli_copula,
    smp.bernoulli_copula_sample,
    prd.bernoulli_predict,
    diagnose.bernoulli_gcopula_diagnose,
)

ZeroInflatedNegBinRegressionSimulator = glm_simulator_generator(
    "ZeroInflatedNegbinRegressionSimulator",
    est.zero_inflated_negbin_regression,
    smp.zero_inflated_negbin_sample,
    prd.zero_inflated_negbin_predict,
    diagnose.zinb_regression_diagnose,
)

ZeroInflatedNegBinCopulaSimulator = glm_simulator_generator(
    "ZeroInflatedNegBinCopulaSimulator",
    est.zero_inflated_negbin_copula,
    smp.zero_inflated_negbin_copula_sample,
    prd.zero_inflated_negbin_predict,
    diagnose.zinb_gcopula_diagnose,
)

ZeroInflatedPoissonRegressionSimulator = glm_simulator_generator(
    "ZeroInflatedNegbinRegressionSimulator",
    est.zero_inflated_poisson_regression,
    smp.zero_inflated_poisson_sample,
    prd.zero_inflated_poisson_predict,
    diagnose.zip_regression_diagnose,
)

GaussianRegressionSimulator = glm_simulator_generator(
    "GaussianRegressionSimulator",
    est.gaussian_regression,
    smp.gaussian_regression_sample,
    prd.gaussian_predict,
    diagnose.gaussian_regression_diagnose
)

GaussianCopulaSimulator = glm_simulator_generator(
    "GaussianCopulaSimulator",
    est.gaussian_copula,
    smp.gaussian_copula_sample,
    prd.gaussian_predict,
    diagnose.gaussian_gcopula_diagnose
)
