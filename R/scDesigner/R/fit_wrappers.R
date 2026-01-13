#' @importFrom reticulate import import_builtins
load_model_helper <- function(pickle_path) {
    cloudpickle <- import("cloudpickle")
    builtins <- import_builtins()
    f <- builtins$open(pickle_path, "rb")
    sim <- cloudpickle$load(f)
    f$close()
    sim
}

parameters_helper <- function(pickle_path) {
    load_model_helper(pickle_path)$parameters
}

#' @importFrom zellkonverter AnnData2SCE
sample_helper <- function(pickle_path, obs) {
    sim <- load_model_helper(pickle_path)
    synthetic_adata <- sim$sample(obs = obs)
    AnnData2SCE(synthetic_adata)
}
