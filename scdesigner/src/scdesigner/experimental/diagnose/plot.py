import scanpy as sc
import numpy as np
import pandas as pd
import altair as alt


def plot_umap(
    adata,
    color=None,
    shape=None,
    facet=None,
    opacity=0.6,
    n_comps=20,
    n_neighbors=15,
    **kwargs
):
    mapping = {"x": "UMAP1", "y": "UMAP2", "color": color, "shape": shape}
    mapping = {k: v for k, v in mapping.items() if v is not None}

    adata_ = adata.copy()
    adata_.X = np.log1p(adata_.X)

    # umap on the top PCA dimensions
    sc.pp.pca(adata_, n_comps=n_comps)
    sc.pp.neighbors(adata_, n_neighbors=n_neighbors, n_pcs=n_comps)
    sc.tl.umap(adata_, **kwargs)

    # get umap embeddings
    umap_df = pd.DataFrame(adata_.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
    umap_df = pd.concat([umap_df, adata_.obs.reset_index(drop=True)], axis=1)

    # encode and visualize
    chart = alt.Chart(umap_df).mark_point(opacity=opacity).encode(**mapping)
    if facet is not None:
        chart = chart.facet(column=alt.Facet(facet))
    return chart


def compare_summary(real, simulated, summary_fun):
    df = pd.DataFrame({"real": summary_fun(real), "simulated": summary_fun(simulated)})

    identity = pd.DataFrame(
        {
            "real": [df["real"].min(), df["real"].max()],
            "simulated": [df["real"].min(), df["real"].max()],
        }
    )
    return alt.Chart(identity).mark_line(color="#dedede").encode(
        x="real", y="simulated"
    ) + alt.Chart(df).mark_circle().encode(x="real", y="simulated")


def check_sparse(X):
    if not isinstance(X, np.ndarray):
        X = X.todense()
    return X


def compare_means(real, simulated, transform=lambda x: x):
    real.X = check_sparse(real.X)
    simulated.X = check_sparse(simulated.X)
    summary = lambda a: np.asarray(transform(a.X).mean(axis=0)).flatten()
    return compare_summary(real, simulated, summary)


def compare_variance(real, simulated, transform=lambda x: x):
    real.X = check_sparse(real.X)
    simulated.X = check_sparse(simulated.X)
    summary = lambda a: np.asarray(np.var(transform(a.X), axis=0)).flatten()
    return compare_summary(real, simulated, summary)


def compare_standard_deviation(real, simulated, transform=lambda x: x):
    real.X = check_sparse(real.X)
    simulated.X = check_sparse(simulated.X)
    summary = lambda a: np.asarray(np.std(transform(a.X), axis=0)).flatten()
    return compare_summary(real, simulated, summary)
