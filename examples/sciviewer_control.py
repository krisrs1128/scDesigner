import altair as alt
import numpy as np
import pandas as pd
import scipy.stats as ss


def cormat(X, projection):
    """
    Correlate all genes with a projection direction
    """
    return np.corrcoef(X.T, projection, rowvar=True)[-1, :-1]


def contrast_scores(X, projection, n_perm=5):
    """
    Compute Contrast Scores

    This follows equation (1) from the clipper paper
    """
    rho_perm = np.zeros((1 + n_perm, X.shape[1]))
    rho_perm[0, :] = cormat(X, projection)

    # correlations across the permutations
    for perm in range(1, n_perm):
        ix = np.random.permutation(len(X))
        rho_perm[perm, :] = cormat(X, projection[ix])

    # compute contrast scores
    contrast = np.zeros(X.shape[1])
    for gene in range(X.shape[1]):
        T = np.sort(abs(rho_perm[:, gene]))[::-1]
        if abs(rho_perm[0, gene]) == T[0]:
            contrast[gene] = T[0] - T[1]
        else:
            contrast[gene] = T[1] - T[0]

    return contrast


def false_discovery_estimate(scores, q=0.05, n_perm=5):
    """
    Identify the FDR Cutoff

    This follows equation (12) from the clipper paper
    """
    sort_ix = np.argsort(scores)[::-1]
    fdr_hat = np.ones(len(scores))

    # compute the FDR estimates
    threshold = None
    for i, ix in enumerate(sort_ix):
        if scores[ix] < 0:
            break

        fdr_i = (1 / n_perm) * (1 + np.sum(scores <= -scores[ix])) / (1 + i)
        fdr_hat[sort_ix[i]] = fdr_i

        if fdr_i < q:
            threshold = ix

    # determine the cutoff
    if threshold is None:
        selection = np.repeat(False, len(sort_ix))
    else:
        selection = scores >= scores[threshold]
    return fdr_hat, selection


def plot_subset(adata, genes, scores):
    """
    Show the raw data associated with a set of selected genes

    Panels are sorted from largest to smallest contrast score (the number in parenthesis)
    """
    gene_data = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs.index)
    gene_data = gene_data.loc[:, genes]
    gene_data.columns = [
        f"""{gene} ({np.round(score, 3)})""" for gene, score in zip(genes, scores)
    ]

    gene_data["projection"] = adata.obs["projection"]
    gene_data = gene_data.melt(
        id_vars=["projection"], var_name="gene", value_name="value"
    )

    return (
        alt.Chart(gene_data)
        .mark_point()
        .encode(x="projection", y="value")
        .facet(facet=alt.Facet("gene:N", sort=genes), columns=5)
        .resolve_scale(y="independent")
        .configure_view(continuousWidth=150, continuousHeight=150)
    )


def test_direction(X, projection):
    rs = np.corrcoef(projection, X, rowvar=False)[0, 1:]

    n = len(X)
    T = -np.abs(rs * np.sqrt(n - 2)) / np.sqrt(1 - (rs**2))
    return {"correlation": rs, "p_value": ss.t.cdf(T, df=n - 2) * 2}


def plot_pval_sequence(online_result, width=600):
    online_result = {k: v for k, v in online_result.items() if k != "tau"}
    online_result = pd.DataFrame(online_result).reset_index()
    online_result["neg_log_pval"] = -np.log(online_result["p_value"])

    return (
        alt.Chart(online_result)
        .mark_circle(size=5)
        .encode(
            x="index:Q",
            y="neg_log_pval:Q",
            color="R:N",
        )
        .properties(width=width)
    )


def plot_alpha_sequence(online_result, width=600):
    online_result = {k: v for k, v in online_result.items() if k != "tau"}
    online_result = pd.DataFrame(online_result).reset_index()

    return (
        alt.Chart(online_result)
        .mark_circle(size=5)
        .encode(x="index:Q", y="alpha_i:Q")
        .properties(width=width)
    )


def lord_test(pval, initial_results=None, gammai=None, alpha=0.05, w0=0.005):
    """"
    This is a translation of "version 1" under:

    https://github.com/bioc/onlineFDR/blob/devel/src/lord.cpp
    
    The only changes are that we don't recompute threhsolds for hypotheses that
    we have already seen. This only necessary because we may continue testing
    for many directions.
    """
    N = len(pval)

    if gammai is None:
        gammai = (
            0.07720838
            * np.log(np.maximum(np.arange(1, N + 2), 2))
            / (np.arange(1, N + 2) * np.exp(np.sqrt(np.log(np.arange(1, N + 2)))))
        )

    # setup variables, substituting previous results if needed
    alphai = np.zeros(N)
    R = np.zeros(N, dtype=bool)
    tau = []
    if initial_results is not None:
        N0 = len(initial_results["p_value"])
        alphai[range(N0)] = initial_results["alpha_i"]
        R[range(N0)] = initial_results["R"]
        tau = initial_results["tau"]
    else:
        N0 = 1
        alphai[0] = gammai[0] * w0
        R[0] = pval[0] <= alphai[0]
        if R[0]:
            tau.append(0)

    # compute lord thresholds iteratively
    K = int(np.sum(R))
    for i in range(N0, N):
        if K <= 1:
            if R[i - 1]:
                tau = [i - 1]
            Cjsum = sum(gammai[i - tau[j] - 1] for j in range(K))
            alphai[i] = w0 * gammai[i] + (alpha - w0) * Cjsum
        else:
            if R[i - 1]:
                tau.append(i - 1)
            tau2 = tau[1:]
            Cjsum = sum(gammai[i - tau2[j] - 1] for j in range(K - 1))
            alphai[i] = (
                w0 * gammai[i] + (alpha - w0) * gammai[i - tau[0] - 1] + alpha * Cjsum
            )

        if pval[i] <= alphai[i]:
            R[i] = True
            K += 1

    return {"p_value": pval, "alpha_i": alphai, "R": R, "tau": tau}
