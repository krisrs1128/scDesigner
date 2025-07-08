from scipy.stats import norm, multivariate_normal
import torch
import numpy as np
import pandas as pd


def marginal_aic_bic(parameters, likelihood, n_sample):
    nparam = len(parameters)
    aic = 2 * nparam - 2 * likelihood
    bic = np.log(n_sample) * nparam - 2 * likelihood
    return aic, bic


def gaussian_copula_aic_bic(loader, parameters, uniformizer):
    covariance = parameters["covariance"]
    with torch.no_grad():
        if not isinstance(covariance, dict):
            nop = (np.sum(covariance != 0) - covariance.shape[0]) / 2
            total_copula_ln = 0
            total_marginal_ln = 0
            n_sample = 0
            for x, y, _ in loader:
                u = uniformizer(parameters, x.cpu().numpy(), y.cpu().numpy())
                z = norm().ppf(u)
                copula_ln = multivariate_normal.logpdf(
                    z, np.zeros(covariance.shape[0]), covariance
                )
                marginal_ln = norm.logpdf(z)
                total_copula_ln += np.sum(copula_ln)
                total_marginal_ln += np.sum(marginal_ln)
                n_sample += z.shape[0]
            aic = -2 * (total_copula_ln - total_marginal_ln) + 2 * nop
            bic = -2 * (total_copula_ln - total_marginal_ln) + np.log(n_sample) * nop
            return aic, bic
        else:
            groups = covariance.keys()
            nop = {
                g: (np.sum(covariance[g] != 0) - covariance[g].shape[0]) / 2
                for g in groups
            }
            for x, y, memberships in loader:
                aic = 0
                bic = 0
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
    return aic, bic


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
    return aic, bic