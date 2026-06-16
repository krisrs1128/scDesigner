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


pancreas = _make_loader("pancreas")
acinar = _make_loader("acinar")
granja_atac = _make_loader("granja_atac")
cite = _make_loader("cite")
embryo = _make_loader("embryo")
marrow = _make_loader("marrow")
mobsp = _make_loader("mobsp")
ovarian = _make_loader("ovarian")
scgemmeth = _make_loader("scgemmeth")
scgemrna = _make_loader("scgemrna")
visium = _make_loader("visium")

__all__ = [
    "DATASETS",
    "FigshareData",
    "pancreas",
    "acinar",
    "granja_atac",
    "cite",
    "embryo",
    "marrow",
    "mobsp",
    "ovarian",
    "scgemmeth",
    "scgemrna",
    "visium",
]
