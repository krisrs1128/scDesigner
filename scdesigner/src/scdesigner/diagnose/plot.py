import scanpy as sc
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt


def adata_df(adata):
    return (
        pd.DataFrame(adata.X, columns=adata.var_names)
        .melt(id_vars=[], value_vars=adata.var_names)
        .reset_index(drop=True)
    )


def merge_samples(adata, sim):
    source = adata_df(adata)
    simulated = adata_df(sim)
    return pd.concat(
        {"real": source, "simulated": simulated}, names=["source"]
    ).reset_index(level="source")


def plot_umap(
    adata,
    color=None,
    shape=None,
    facet=None,
    opacity=0.6,
    n_comps=20,
    n_neighbors=15,
    transform=lambda x: np.log1p(x),
    **kwargs
):
    mapping = {"x": "UMAP1", "y": "UMAP2", "color": color, "shape": shape}
    mapping = {k: v for k, v in mapping.items() if v is not None}

    adata_ = adata.copy()
    adata_.X = check_sparse(adata_.X)
    Z = transform(adata_.X)
    if Z.shape[1] == adata_.X.shape[1]:
        adata_.X = transform(adata_.X)
    else:
        adata_ = adata_[:, : Z.shape[1]]
        adata_.X = Z
        adata_.var_names = [f"transform_{k}" for k in range(Z.shape[1])]

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


def plot_pca(
    adata,
    color=None,
    shape=None,
    facet=None,
    opacity=0.6,
    plot_dims=[0, 1],
    transform=lambda x: np.log1p(x),
    **kwargs
):
    mapping = {"x": "PCA1", "y": "PCA2", "color": color, "shape": shape}
    mapping = {k: v for k, v in mapping.items() if v is not None}

    adata_ = adata.copy()
    adata_.X = check_sparse(adata_.X)
    adata_.X = transform(adata_.X)

    # get PCA scores
    sc.pp.pca(adata_, **kwargs)
    pca_df = pd.DataFrame(adata_.obsm["X_pca"][:, plot_dims], columns=["PCA1", "PCA2"])
    pca_df = pd.concat([pca_df, adata_.obs.reset_index(drop=True)], axis=1)

    # plot
    chart = alt.Chart(pca_df).mark_point(opacity=opacity).encode(**mapping)
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
    real_, simulated_ = prepare_dense(real, simulated)
    summary = lambda a: np.asarray(transform(a.X).mean(axis=0)).flatten()
    return compare_summary(real_, simulated_, summary)


def prepare_dense(real, simulated):
    real_ = real.copy()
    simulated_ = simulated.copy()
    real_.X = check_sparse(real_.X)
    simulated_.X = check_sparse(simulated_.X)
    return real_, simulated_


def compare_variances(real, simulated, transform=lambda x: x):
    real_, simulated_ = prepare_dense(real, simulated)
    summary = lambda a: np.asarray(np.var(transform(a.X), axis=0)).flatten()
    return compare_summary(real_, simulated_, summary)


def compare_standard_deviation(real, simulated, transform=lambda x: x):
    real_, simulated_ = prepare_dense(real, simulated)
    summary = lambda a: np.asarray(np.std(transform(a.X), axis=0)).flatten()
    return compare_summary(real_, simulated_, summary)


def concat_real_sim(real, simulated):
    real_, simulated_ = prepare_dense(real, simulated)
    real_.obs["source"] = "real"
    simulated_.obs["source"] = "simulated"
    return real_.concatenate(simulated_, join="outer", batch_key=None)


def compare_umap(real, simulated, transform=lambda x: x, **kwargs):
    adata = concat_real_sim(real, simulated)
    return plot_umap(adata, facet="source", transform=transform, **kwargs)


def compare_pca(real, simulated, transform=lambda x: x, **kwargs):
    adata = concat_real_sim(real, simulated)
    return plot_pca(adata, facet="source", transform=transform, **kwargs)


def plot_hist(sim_data, real_data, idx):
    sim = sim_data[:, idx]
    real = real_data[:, idx]
    b = np.linspace(min(min(sim), min(real)), max(max(sim), max(real)), 50)

    plt.hist([real, sim], b, label=["Real", "Simulated"], histtype="bar")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def compare_ecdf(adata, sim, var_names=None, max_plot=10, n_cols=5, **kwargs):
    if var_names is None:
        var_names = adata.var_names[:max_plot]

    combined = merge_samples(adata[:, var_names], sim.sample()[:, var_names])
    alt.data_transformers.enable("vegafusion")

    plot = (
        alt.Chart(combined)
        .transform_window(
            ecdf="cume_dist()", sort=[{"field": "value"}], groupby=["variable"]
        )
        .mark_line(
            interpolate="step-after",
        )
        .encode(
            x="value:Q",
            y="ecdf:Q",
            color="source:N",
            facet=alt.Facet(
                "variable", sort=alt.EncodingSortField("value"), columns=n_cols
            ),
        )
        .properties(**kwargs)
    )
    plot.show()
    return plot, combined


def compare_boxplot(adata, sim, var_names=None, max_plot=20, **kwargs):
    if var_names is None:
        var_names = adata.var_names[:max_plot]

    combined = merge_samples(adata[:, var_names], sim.sample()[:, var_names])
    alt.data_transformers.enable("vegafusion")

    plot = (
        alt.Chart(combined)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("value:Q").scale(zero=False),
            y=alt.Y(
                "variable:N",
                sort=alt.EncodingSortField("mid_box_value", order="descending"),
            ),
            facet="source:N",
        )
        .properties(**kwargs)
    )
    plot.show()
    return plot, combined


def compare_histogram(adata, sim, var_names=None, max_plot=20, n_cols=5, **kwargs):
    if var_names is None:
        var_names = adata.var_names[:max_plot]

    combined = merge_samples(adata[:, var_names], sim.sample()[:, var_names])
    alt.data_transformers.enable("vegafusion")

    plot = (
        alt.Chart(combined)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("value:Q").bin(maxbins=20),
            y=alt.Y("count()").stack(None),
            color="source:N",
            facet=alt.Facet(
                "variable", sort=alt.EncodingSortField("bin_maxbins_20_value")
            ),
        )
        .properties(**kwargs)
    )
    plot.show()
    return plot, combined
