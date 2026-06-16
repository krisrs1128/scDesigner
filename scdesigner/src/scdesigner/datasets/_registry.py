"""Registry of example datasets. Add one by appending to :data:`DATASETS`."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FigshareData:
    """A dataset hosted on Figshare.

    ``name`` is the registry key and cache filename stem; ``file_id`` is the
    numeric id at the end of a Figshare download URL.
    """

    name: str
    file_id: int


DATASETS: dict[str, FigshareData] = {
    "pancreas": FigshareData(name="pancreas", file_id=60087086),
    "acinar": FigshareData(name="acinar", file_id=64417734),
    "granja_atac": FigshareData(name="granja_atac", file_id=64418073),
    "cite": FigshareData(name="cite", file_id=65593449),
    "embryo": FigshareData(name="embryo", file_id=65593671),
    "marrow": FigshareData(name="marrow", file_id=65593950),
    "mobsp": FigshareData(name="mobsp", file_id=65594652),
    "ovarian": FigshareData(name="ovarian", file_id=65593941),
    "scgemmeth": FigshareData(name="scgemmeth", file_id=65594694),
    "scgemrna": FigshareData(name="scgemrna", file_id=65594697),
    "visium": FigshareData(name="visium", file_id=65594991),
}
