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
    "plot_umap",
    "plot_pca",
    "plot_hist",
    "compare_means",
    "compare_variances",
    "compare_standard_deviation",
    "compare_umap",
    "compare_pca",
    "negbin_regression_diagnose",
    "negbin_gcopula_diagnose",
    "poisson_regression_diagnose",
    "poisson_gcopula_diagnose",
    "bernoulli_regression_diagnose",
    "bernoulli_gcopula_diagnose",
    "zinb_regression_diagnose",
    "zinb_gcopula_diagnose",
    "zip_regression_diagnose",
]


###############################################################################
## Methods for calculating marginal/gaussian copula AIC/BIC
###############################################################################

negbin_regression_diagnose = compose_marginal_diagnose(est.negbin.negbin_regression_likelihood)
negbin_gcopula_diagnose = compose_gcopula_diagnose(est.negbin.negbin_regression_likelihood,
                                                   est.negbin.negbin_uniformizer)
poisson_regression_diagnose = compose_marginal_diagnose(est.poisson.poisson_regression_likelihood)
poisson_gcopula_diagnose = compose_gcopula_diagnose(est.poisson.poisson_regression_likelihood,
                                                    est.poisson.poisson_uniformizer)
bernoulli_regression_diagnose = compose_marginal_diagnose(est.bernoulli.bernoulli_regression_likelihood)
bernoulli_gcopula_diagnose = compose_gcopula_diagnose(est.bernoulli.bernoulli_regression_likelihood,
                                                      est.bernoulli.bernoulli_uniformizer)
zinb_regression_diagnose = compose_marginal_diagnose(est.zero_inflated_negbin.zero_inflated_negbin_regression_likelihood)
zinb_gcopula_diagnose = compose_gcopula_diagnose(est.zero_inflated_negbin.zero_inflated_negbin_regression_likelihood,
                                                 est.zero_inflated_negbin.zero_inflated_negbin_uniformizer)
zip_regression_diagnose = compose_marginal_diagnose(est.zero_inflated_poisson.zero_inflated_poisson_regression_likelihood)
