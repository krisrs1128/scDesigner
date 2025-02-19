import scanpy as sc
import numpy as np
import pandas as pd
import altair as alt

def plot_umap(adata, color=None, shape=None, opacity=0.6, **kwargs):
    mapping = {"x": "UMAP1", "y": "UMAP2", "color": color, "shape": shape}
    mapping = {k: v for k, v in mapping.items() if v is not None}

    adata_ = adata.copy()
    adata_.X = np.log1p(adata_.X)
    sc.pp.neighbors(adata_)
    sc.tl.umap(adata_, **kwargs)

    # get umap embeddings
    umap_df = pd.DataFrame(adata_.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
    umap_df = pd.concat([umap_df, adata_.obs.reset_index(drop=True)], axis=1)

    # encode and visualize
    return alt.Chart(umap_df).mark_point(opacity=opacity).encode(**mapping)