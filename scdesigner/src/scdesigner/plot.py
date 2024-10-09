import altair as alt
import anndata as ad
import pandas as pd
import scanpy as sc

def embedding(sim, adata=None, tool="umap", **kwargs):
    exper = sim
    if adata is not None:
        exper = ad.concat([exper, sim], label="simulated")

    if tool == "umap":
        sc.pp.neighbors(exper)

    # prepare data for the visualization
    getattr(sc.tl, tool)(exper, **kwargs)
    plot_data = pd.concat([
        pd.DataFrame(
            exper.obsm["X_umap"], 
            columns=[f"{tool}_{i}" for i in range(exper.obsm["X_umap"].shape[1])]
        ),
        exper.obs.reset_index()
    ], axis=1)

    # generate the plot
    plot = alt.Chart(plot_data).mark_circle(opacity=0.6).encode(
        x=f"{tool}_0",
        y=f"{tool}_1",
        color="simulated" if adata is not None else alt.ColorValue("black")
    )
    plot.show()
    return plot, exper

def adata_df(adata):
    return pd.DataFrame(adata.X, columns=adata.var_names)\
        .melt(id_vars=[], value_vars=adata.var_names)\
        .reset_index(drop=True)

def merge_samples(adata, sim):
    source = adata_df(adata)
    simulated = adata_df(sim)
    return pd.concat({ "real": source, "simulated": simulated }, names=["source"])\
        .reset_index(level="source")


def ecdf(adata, sim, var_names=None, max_plot=10, n_cols=5, **kwargs):
    if var_names is None:
        var_names = adata.var_names[:max_plot]

    combined = merge_samples(adata[:, var_names], sim.sample()[:, var_names])
    alt.data_transformers.enable("vegafusion")

    plot = alt.Chart(combined).transform_window(
        ecdf="cume_dist()",
        sort=[{"field": "value"}],
        groupby=["variable"]
    ).mark_line(
        interpolate="step-after",
    ).encode(
        x="value:Q",
        y="ecdf:Q",
        color="source:N",
        facet=alt.Facet("variable", sort=alt.EncodingSortField("value"), columns=n_cols)
    ).properties(**kwargs)
    plot.show()
    return plot, combined