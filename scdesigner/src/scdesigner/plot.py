import altair
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
    plot = altair.Chart(plot_data).mark_circle(opacity=0.6).encode(
        x=f"{tool}_0",
        y=f"{tool}_1",
        color="simulated" if adata is not None else altair.ColorValue("black")
    )
    plot.show()
    return plot, exper

    



