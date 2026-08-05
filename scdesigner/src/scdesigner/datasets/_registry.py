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
    "acinar": FigshareData(name="acinar", file_id=64417734),
    "batch": FigshareData(name="batch", file_id=66053981),
    "cite": FigshareData(name="cite", file_id=65593449),
    "embryo": FigshareData(name="embryo", file_id=65593671),
    "granja_atac": FigshareData(name="granja_atac", file_id=64418073),
    "gyrus": FigshareData(name="gyrus", file_id=66126302),
    "hvg_embryo_atlas": FigshareData(name="hvg_embryo_atlas", file_id=66126275),
    "ifnb": FigshareData(name="ifnb", file_id=66126296),
    "marrow": FigshareData(name="marrow", file_id=65593950),
    "mobsc": FigshareData(name="mobsc", file_id=66126281),
    "mobsp": FigshareData(name="mobsp", file_id=65594652),
    "mobspmix": FigshareData(name="mobspmix", file_id=66126278),
    "mouse_cortex": FigshareData(name="mouse_cortex", file_id=67164080),
    "mouse_visual": FigshareData(name="mouse_visual", file_id=67164077),
    "ovarian": FigshareData(name="ovarian", file_id=65593941),
    "pancreas": FigshareData(name="pancreas", file_id=60087086),
    "prostate": FigshareData(name="prostate", file_id=66126284),
    "scgemmeth": FigshareData(name="scgemmeth", file_id=65594694),
    "scgemrna": FigshareData(name="scgemrna", file_id=65594697),
    "sciatac": FigshareData(name="sciatac", file_id=66126290),
    "sciatac_back": FigshareData(name="sciatac_back", file_id=66126293),
    "sciatac_fore": FigshareData(name="sciatac_fore", file_id=66126287),
    "seqfish": FigshareData(name="seqfish", file_id=66126272),
    "slide": FigshareData(name="slide", file_id=66126299),
    "visium": FigshareData(name="visium", file_id=65594991),
    "zhengmix4": FigshareData(name="zhengmix4", file_id=67164083),
}
