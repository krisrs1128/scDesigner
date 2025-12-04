from pathlib import Path
from typing import Optional, Union
import anndata
import joblib
import os
import urllib.request

ARCHIVE_URL = "https://figshare.com/ndownloader/files/60087086"


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
        urllib.request.urlretrieve(ARCHIVE_URL, str(tmp_path))
        adata = anndata.read_h5ad(str(tmp_path))
        joblib.dump(adata, str(cache_path), compress=6)
        return adata
    except:
        pass
    if tmp_path.exists():
        tmp_path.unlink()