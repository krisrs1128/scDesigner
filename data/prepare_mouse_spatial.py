#!/usr/bin/env python3
"""Download two SpatialBenchmarking datasets and save them as H5AD."""

from pathlib import Path
import sys
import tempfile
from urllib.request import urlretrieve

import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix


REPOSITORY = "https://github.com/QuKunLab/SpatialBenchmarking"
RAW_URL = "https://raw.githubusercontent.com/QuKunLab/SpatialBenchmarking/main"

DATASETS = {
    "mouse_cortex": {
        "counts": "FigureData/Figure4/Dataset4_seqFISH%2B/Rawdata/Spatial_count.txt",
        "coordinates": "FigureData/Figure4/Dataset4_seqFISH%2B/Rawdata/Locations_seqfish.txt",
        "technology": "seqFISH+",
    },
    "mouse_visual": {
        "counts": "FigureData/Figure4/Dataset10_STARmap/Rawdata/Spatial_count.txt",
        "coordinates": "FigureData/Figure4/Dataset10_STARmap/Rawdata/Locations.txt",
        "technology": "STARmap",
    },
}


def prepare_dataset(name, config, raw_dir, output_dir):
    counts_path = raw_dir / f"{name}_counts.txt"
    coordinates_path = raw_dir / f"{name}_coordinates.txt"
    urlretrieve(f"{RAW_URL}/{config['counts']}", counts_path)
    urlretrieve(f"{RAW_URL}/{config['coordinates']}", coordinates_path)

    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    coordinates = pd.read_csv(coordinates_path, sep="\t")

    counts.index = counts.index.astype(str)
    spatial = coordinates[["X", "Y"]].to_numpy()
    obs = pd.DataFrame(
        {"spatial1": spatial[:, 0], "spatial2": spatial[:, 1]},
        index=counts.index,
    )
    adata = ad.AnnData(
        X=csr_matrix(counts.to_numpy()),
        obs=obs,
        var=pd.DataFrame(index=counts.columns),
    )
    adata.obsm["spatial"] = spatial
    adata.uns.update(
        {
            "source": "QuKunLab/SpatialBenchmarking",
            "source_repository": REPOSITORY,
            "expression_source_file": config["counts"].replace("%2B", "+"),
            "coordinate_source_file": config["coordinates"].replace("%2B", "+"),
            "technology": config["technology"],
            "coordinate_alignment": "row order",
        }
    )

    output_path = output_dir / f"{name}.h5ad"
    adata.write_h5ad(output_path, compression="gzip")
    print(f"Wrote {output_path}: {adata.n_obs} cells x {adata.n_vars} genes")


def main():
    output_dir = Path(__file__).parent

    with tempfile.TemporaryDirectory() as raw_dir:
        for name, config in DATASETS.items():
            prepare_dataset(name, config, Path(raw_dir), output_dir)


if __name__ == "__main__":
    main()
