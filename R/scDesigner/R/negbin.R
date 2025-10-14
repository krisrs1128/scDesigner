

#' @importFrom basilisk basiliskStart basiliskRun basiliskStop
#' @export
test_sim <- function() {
    cl <- basiliskStart(env)
    example_call <- basiliskRun(cl, \() {
        x <- reticulate::import("scdesigner.minimal")
        names(x)
    })
    basiliskStop(cl)
}