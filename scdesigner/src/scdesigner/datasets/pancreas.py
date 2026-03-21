from pathlib import Path
from typing import Optional, Union
import anndata
import joblib
import os
import urllib.request

FIGSHARE_FILE_ID = 60087086
ARCHIVE_URL = f"https://api.figshare.com/v2/file/download/{FIGSHARE_FILE_ID}"


def _ensure_data_home(data_home: Optional[Union[str, os.PathLike]]) -> Path:
    base = Path(data_home) if data_home is not None else Path.home() / ".scdesigner_data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def fetch_pancreas(
    *,
    data_home: Optional[Union[str, os.PathLike]] = None,
    download_if_missing: bool = True,
) -> Optional[object]:
    data_home_path = _ensure_data_home(data_home)
    cache_path = data_home_path / "pancreas.joblib"
    if cache_path.exists():
        return joblib.load(cache_path)

    if not download_if_missing:
        return None

    tmp_path = data_home_path / "pancreas.h5ad"
    try:
        req = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "scdesigner/1.0"})
        with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as f:
            f.write(response.read())
        adata = anndata.read_h5ad(str(tmp_path))
        joblib.dump(adata, str(cache_path), compress=6)
        return adata
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download Figshare {FIGSHARE_FILE_ID}: {e}") from e