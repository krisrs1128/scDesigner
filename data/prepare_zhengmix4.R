#!/usr/bin/env Rscript

# Usage: Rscript data/prepare_zhengmix4.R [output-directory]

suppressPackageStartupMessages({
  library(DuoClustering2018)
  library(SingleCellExperiment)
  library(zellkonverter)
})

sce <- sce_filteredExpr10_Zhengmix4eq(metadata = FALSE)

gene_symbols <- as.character(rowData(sce)$symbol)
rownames(sce) <- gene_symbols

colData(sce)$cell_type <- factor(colData(sce)$phenoid)

# Preserve the annotations, but write only raw counts as the expression matrix.
output_sce <- SingleCellExperiment(
  assays = list(counts = counts(sce)),
  rowData = rowData(sce),
  colData = colData(sce),
  metadata = metadata(sce)
)
metadata(output_sce)$source <- "DuoClustering2018"
metadata(output_sce)$source_object <- "sce_filteredExpr10_Zhengmix4eq"
metadata(output_sce)$dataset <- "Zhengmix4eq"
metadata(output_sce)$original_source <- "Zheng et al. (2017), 10x Genomics"

writeH5AD(output_sce, "zhengmix4.h5ad", X_name = "counts", compression = "gzip")

cat(sprintf(
  "Wrote %s: %d cells x %d genes\n",
  "zhengmix4.h5ad", ncol(output_sce), nrow(output_sce)
))
