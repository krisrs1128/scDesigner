import altair as alt
import numpy as np
import pandas as pd


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
