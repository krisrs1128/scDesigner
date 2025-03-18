import scanpy as sc
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt


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


def plot_hist(sim_data, real_data, idx):
    sim = sim_data[:, idx]
    real = real_data[:, idx]
    b = np.linspace(min(min(sim), min(real)), max(max(sim), max(real)), 50)

    plt.hist([real, sim], b, label=["Real", "Simulated"], histtype="bar")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
