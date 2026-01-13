#' Clear the basilisk environment cache
#'
#' Run this function after updating scDesigner to force basilisk to rebuild
#' the Python environment with the new version.
#'
#' @export
clear_basilisk_cache <- function() {
    basilisk_dir <- basilisk::basiliskCacheDir()
    pkg_dir <- file.path(basilisk_dir, "scDesigner")

    if (dir.exists(pkg_dir)) {
        unlink(pkg_dir, recursive = TRUE)
        message("Cleared basilisk cache for scDesigner. The environment will be rebuilt on next use.")
    } else {
        message("No basilisk cache found for scDesigner.")
    }
}
