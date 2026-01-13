#' @importFrom reticulate import import_builtins
#' @importFrom zellkonverter SCE2AnnData
fit_helper <- function(sce, mean_formula, dispersion_formula, copula_formula, pickle_path, ...) {
    scd <- import("scdesigner.simulators")
    cloudpickle <- import("cloudpickle")
    builtins <- import_builtins()
    adata <- SCE2AnnData(sce)
    sim <- scd$NegBinCopula(mean_formula, dispersion_formula, copula_formula)
    sim$fit(
        adata,
        ...
    )
    f <- builtins$open(pickle_path, "wb")
    cloudpickle$dump(sim, f)
    f$close()
    NULL
}


#' Negative Binomial Copula Simulator
#' @importFrom basilisk basiliskStart basiliskRun basiliskStop
#' @export
NegBinCopula <- R6::R6Class(
    "NegBinCopula",
    public = list(
        mean_formula = NULL,
        dispersion_formula = NULL,
        copula_formula = NULL,
        pickle_path = NULL,
        basilisk_proc = NULL,
        initialize = function(mean_formula = NULL, dispersion_formula = NULL, copula_formula = NULL) {
            self$mean_formula <- mean_formula
            self$dispersion_formula <- dispersion_formula
            self$copula_formula <- copula_formula
            self$pickle_path <- tempfile(fileext = ".pkl")
        },
        fit = function(sce, max_epochs = 50L, lr = 0.1, batch_size = 1024L) {
            self$basilisk_proc <- basiliskStart(env)
            args <- list(
                self$basilisk_proc,
                fit_helper,
                sce = sce,
                mean_formula = self$mean_formula,
                dispersion_formula = self$dispersion_formula,
                copula_formula = self$copula_formula,
                pickle_path = self$pickle_path,
                max_epochs = as.integer(max_epochs),
                batch_size = as.integer(batch_size)
            )

            do.call(basiliskRun, args)
            invisible(self)
        },
        parameters = function() {
            if (is.null(self$basilisk_proc)) self$basilisk_proc <- basiliskStart(env)
            basiliskRun(self$basilisk_proc, parameters_helper, pickle_path = self$pickle_path)
        },
        sample = function(obs = NULL) {
            if (is.null(self$basilisk_proc)) self$basilisk_proc <- basiliskStart(env)
            basiliskRun(self$basilisk_proc, sample_helper, pickle_path = self$pickle_path, obs = obs)
        },
        stop = function() {
            if (!is.null(self$basilisk_proc)) {
                basiliskStop(self$basilisk_proc)
                self$basilisk_proc <- NULL
            }
            invisible(self)
        }
    ),
    private = list(
        finalize = function() {
            if (!is.null(self$basilisk_proc)) basiliskStop(self$basilisk_proc)
            if (!is.null(self$pickle_path) && file.exists(self$pickle_path)) unlink(self$pickle_path)
        }
    )
)
