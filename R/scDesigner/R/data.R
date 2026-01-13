ARCHIVE_URL <- "https://figshare.com/ndownloader/files/60087086"

ensure_data_home <- function(data_home = NULL) {
    base <- if (is.null(data_home)) file.path(path.expand("~"), ".scdesigner_data") else data_home
    if (!dir.exists(base)) dir.create(base, recursive = TRUE)
    base
}

download_and_cache <- function(url, tmp_path, cache_path) {
    download.file(url, tmp_path, mode = "wb", quiet = TRUE)
    adata <- zellkonverter::readH5AD(tmp_path)
    saveRDS(adata, cache_path, compress = "xz")
    adata
}

#' Fetch pancreas dataset
#'
#' @param data_home Optional path to directory for storing data
#' @param download_if_missing Whether to download if data not found locally
#' @return SingleCellExperiment object or NULL if not found and download_if_missing=FALSE
#' @export
fetch_pancreas <- function(data_home = NULL, download_if_missing = TRUE) {
    data_home_path <- ensure_data_home(data_home)
    cache_path <- file.path(data_home_path, "pancreas.rds")

    if (file.exists(cache_path)) {
        return(readRDS(cache_path))
    }

    if (!download_if_missing) {
        return(NULL)
    }

    tmp_path <- file.path(data_home_path, "pancreas.h5ad")
    tryCatch({
        download_and_cache(ARCHIVE_URL, tmp_path, cache_path)
    }, error = function(e) NULL, finally = {
        if (file.exists(tmp_path)) file.remove(tmp_path)
    })
}
