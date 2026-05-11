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
    "pancreas":    FigshareData(name="pancreas",    file_id=60087086),
    "acinar":      FigshareData(name="acinar",      file_id=64417734),
    "granja_atac": FigshareData(name="granja_atac", file_id=64418073),
}
