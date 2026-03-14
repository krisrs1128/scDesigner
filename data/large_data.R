# https://bioconductor.org/packages/release/data/experiment/vignettes/MouseGastrulationData/inst/doc/MouseGastrulationData.html

if(!require("MouseGastrulationData")) BiocManager::install("MouseGastrulationData")

library(MouseGastrulationData)
library(scran)
library(zellkonverter)

sce <- EmbryoAtlasData()

# LogNorm-transformation and select top 10000 HVGs
sce <- logNormCounts(sce)
dec <- modelGeneVar(sce)
hvgs <- getTopHVGs(dec, n = 10000)
sce_hvg <- sce[hvgs, ]

# We can choose to remove technical artefacts
singlets <- which(!(colData(sce_hvg)$doublet | colData(sce_hvg)$stripped))
sce_hvg <- sce_hvg[, singlets]

writeH5AD(sce_hvg, paste0("data/HVG_embryoatlas.h5ad"))