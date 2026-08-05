"""Example datasets for scdesigner tutorials.

Fetched on demand from Figshare and cached under ``$SCDESIGNER_DATA``
(or ``~/.scdesigner_data``); override per call with ``data_home=``.

    from scdesigner import datasets
    adata = datasets.pancreas()

See :data:`DATASETS` for the full list.
"""

import os
import anndata

from ._figshare import fetch_figshare_h5ad
from ._registry import DATASETS, FigshareData


def _make_loader(name: str):
    spec = DATASETS[name]

    def loader(
        *,
        data_home: str | os.PathLike | None = None,
        download_if_missing: bool = True,
    ) -> anndata.AnnData | None:
        return fetch_figshare_h5ad(
            figshare_file_id=spec.file_id,
            cache_name=f"{spec.name}.h5ad",
            data_home=data_home,
            download_if_missing=download_if_missing,
        )

    loader.__name__ = name
    loader.__qualname__ = name
    loader.__doc__ = f"Load the {name!r} dataset."
    return loader


acinar = _make_loader("acinar")
batch = _make_loader("batch")
cite = _make_loader("cite")
embryo = _make_loader("embryo")
granja_atac = _make_loader("granja_atac")
gyrus = _make_loader("gyrus")
hvg_embryo_atlas = _make_loader("hvg_embryo_atlas")
ifnb = _make_loader("ifnb")
marrow = _make_loader("marrow")
mobsc = _make_loader("mobsc")
mobsp = _make_loader("mobsp")
mobspmix = _make_loader("mobspmix")
mouse_cortex = _make_loader("mouse_cortex")
mouse_visual = _make_loader("mouse_visual")
ovarian = _make_loader("ovarian")
pancreas = _make_loader("pancreas")
prostate = _make_loader("prostate")
scgemmeth = _make_loader("scgemmeth")
scgemrna = _make_loader("scgemrna")
sciatac = _make_loader("sciatac")
sciatac_back = _make_loader("sciatac_back")
sciatac_fore = _make_loader("sciatac_fore")
seqfish = _make_loader("seqfish")
slide = _make_loader("slide")
visium = _make_loader("visium")
zhengmix4 = _make_loader("zhengmix4")

__all__ = [
    "DATASETS",
    "FigshareData",
    "acinar",
    "batch",
    "cite",
    "embryo",
    "granja_atac",
    "gyrus",
    "hvg_embryo_atlas",
    "ifnb",
    "marrow",
    "mobsc",
    "mobsp",
    "mobspmix",
    "mouse_cortex",
    "mouse_visual",
    "ovarian",
    "pancreas",
    "prostate",
    "scgemmeth",
    "scgemrna",
    "sciatac",
    "sciatac_back",
    "sciatac_fore",
    "seqfish",
    "slide",
    "visium",
    "zhengmix4",
]
