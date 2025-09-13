from .plot import (
    plot_umap,
    plot_hist,
    compare_means,
    compare_variances,
    compare_standard_deviation,
    compare_umap,
    compare_pca,
)
from .aic_bic import compose_marginal_diagnose, compose_gcopula_diagnose
from .. import estimators as est

__all__ = [
    "bernoulli_gcopula_diagnose",
    "bernoulli_regression_diagnose",
    "compare_means",
    "compare_pca",
    "compare_standard_deviation",
    "compare_umap",
    "compare_variances",
    "gaussian_regression_diagnose",
    "negbin_gcopula_diagnose",
    "negbin_regression_diagnose",
    "plot_hist",
    "plot_pca",
    "plot_umap",
    "poisson_gcopula_diagnose",
    "poisson_regression_diagnose",
    "zinb_gcopula_diagnose",
    "zinb_regression_diagnose",
    "zip_regression_diagnose"
]


###############################################################################
## Methods for calculating marginal/gaussian copula AIC/BIC
###############################################################################

negbin_regression_diagnose = compose_marginal_diagnose(est.negbin.negbin_regression_likelihood,
                                                       allowed_keys=['mean', 'dispersion'])
negbin_gcopula_diagnose = compose_gcopula_diagnose(est.negbin.negbin_regression_likelihood,
                                                   est.negbin.negbin_uniformizer,
                                                   allowed_keys=['mean', 'dispersion'])
poisson_regression_diagnose = compose_marginal_diagnose(est.poisson.poisson_regression_likelihood,
                                                        allowed_keys=['mean'])
poisson_gcopula_diagnose = compose_gcopula_diagnose(est.poisson.poisson_regression_likelihood,
                                                    est.poisson.poisson_uniformizer,
                                                    allowed_keys=['mean'])
bernoulli_regression_diagnose = compose_marginal_diagnose(est.bernoulli.bernoulli_regression_likelihood,
                                                          allowed_keys=['mean'])
bernoulli_gcopula_diagnose = compose_gcopula_diagnose(est.bernoulli.bernoulli_regression_likelihood,
                                                      est.bernoulli.bernoulli_uniformizer,
                                                      allowed_keys=['mean'])
zinb_regression_diagnose = compose_marginal_diagnose(est.zero_inflated_negbin.zero_inflated_negbin_regression_likelihood,
                                                     allowed_keys=['mean', 'dispersion', 'zero_inflation'])
zinb_gcopula_diagnose = compose_gcopula_diagnose(est.zero_inflated_negbin.zero_inflated_negbin_regression_likelihood,
                                                 est.zero_inflated_negbin.zero_inflated_negbin_uniformizer,
                                                 allowed_keys=['mean', 'dispersion', 'zero_inflation'])
zip_regression_diagnose = compose_marginal_diagnose(est.zero_inflated_poisson.zero_inflated_poisson_regression_likelihood,
                                                    allowed_keys=['mean', 'zero_inflation'])
gaussian_regression_diagnose = compose_marginal_diagnose(est.gaussian.gaussian_regression_likelihood,
                                                         allowed_keys=['mean', 'sdev'])
gaussian_gcopula_diagnose = compose_gcopula_diagnose(est.gaussian.gaussian_regression_likelihood, 
                                                     est.gaussian.gaussian_uniformizer,
                                                     allowed_keys=['mean', 'sdev'])
