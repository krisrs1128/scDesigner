"""Low-level Figshare download helper. Caches files as ``.h5ad`` under
``$SCDESIGNER_DATA`` (or ``~/.scdesigner_data``).
"""

from pathlib import Path
import anndata
import os
import urllib.request


def ensure_data_home(data_home: str | os.PathLike | None) -> Path:
    """Return the cache directory, creating it if needed.

    Resolution order: ``data_home`` arg, ``$SCDESIGNER_DATA``, then
    ``~/.scdesigner_data``.
    """
    if data_home is None:
        data_home = os.environ.get("SCDESIGNER_DATA") or Path.home() / ".scdesigner_data"
    base = Path(data_home)
    base.mkdir(parents=True, exist_ok=True)
    return base


def fetch_figshare_h5ad(
    *,
    figshare_file_id: int,
    cache_name: str,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
) -> anndata.AnnData | None:
    """Return a cached AnnData, downloading from Figshare on a cache miss.

    Returns ``None`` if uncached and ``download_if_missing`` is False.
    """
    cache_path = ensure_data_home(data_home) / cache_name
    if cache_path.exists():
        return anndata.read_h5ad(cache_path)
    if not download_if_missing:
        return None

    url = f"https://api.figshare.com/v2/file/download/{figshare_file_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "scdesigner"})
    try:
        with urllib.request.urlopen(req) as response, open(cache_path, "wb") as f:
            f.write(response.read())
        return anndata.read_h5ad(cache_path)
    except Exception as e:
        cache_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download Figshare {figshare_file_id}: {e}") from e
