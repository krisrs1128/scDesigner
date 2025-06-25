from scipy.stats import norm, multivariate_normal
import numpy as np
import pandas as pd


def marginal(parameters, likelihood, y):
    # nparam = 0
    # if not isinstance(parameters, dict):
    #     nparam = parameters.count()
    # else:
    #     for key in parameters.keys():
    #         if key is not "covariance":
    #             nparam += parameters[key].count().to_numpy()
    nparam = len(parameters)
    aic = 2 * nparam - 2 * likelihood
    bic = np.log(y.shape[0]) * nparam - 2 * likelihood
    return aic, bic  # pd.DataFrame({"marginal AIC": aic, "marginal BIC": bic})


def gaussian_copula(covariance, uniformizer, memberships=None):
    u = uniformizer
    if not isinstance(covariance, dict):
        nop = (np.sum(covariance != 0) - covariance.shape[0]) / 2
        z = norm().ppf(u)
        copula_ln = multivariate_normal.logpdf(
            z, np.zeros(covariance.shape[0]), covariance
        )
        marginal_ln = norm.logpdf(z)
        aic = -2 * (np.sum(copula_ln) - np.sum(marginal_ln)) + 2 * nop
        bic = -2 * (np.sum(copula_ln) - np.sum(marginal_ln)) + np.log(z.shape[0]) * nop
    else:
        groups = covariance.keys()
        nop = {
            g: (np.sum(covariance[g] != 0) - covariance[g].shape[0]) / 2 for g in groups
        }
        aic = 0  # aic = {g: 0 for g in groups}
        bic = 0  # bic = {g: 0 for g in groups}
        for g in groups:
            ix = np.where(np.array(memberships) == g)
            z = norm().ppf(u[ix])
            copula_ln = multivariate_normal.logpdf(
                z, np.zeros(covariance[g].shape[0]), covariance[g]
            )
            marginal_ln = norm.logpdf(z)
            aic += -2 * (np.sum(copula_ln) - np.sum(marginal_ln)) + 2 * nop[g]
            bic += (
                -2 * (np.sum(copula_ln) - np.sum(marginal_ln))
                + np.log(z.shape[0]) * nop[g]
            )
    return aic, bic  # (
    #     pd.DataFrame({"copula AIC": aic, "copula BIC": bic}, index=groups)
    #     if isinstance(covariance, dict)
    #     else pd.Series({"copula AIC": aic, "copula BIC": bic})
    # )
