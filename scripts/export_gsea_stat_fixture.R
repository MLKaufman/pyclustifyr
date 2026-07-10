#!/usr/bin/env Rscript
# Regenerates tests/data/gsea_stats_{vals,names}.txt used to validate
# calc_gsea_stat() against fgsea:::calcGseaStat() exactly.

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
repo_root <- normalizePath(file.path(dirname(script_path), ".."))
out_dir <- file.path(repo_root, "tests", "data")

set.seed(7)
stats <- setNames(sort(rnorm(50), decreasing = TRUE), paste0("g", 1:50))
writeLines(as.character(stats), file.path(out_dir, "gsea_stats_vals.txt"))
writeLines(names(stats), file.path(out_dir, "gsea_stats_names.txt"))

cat("Exported GSEA stat fixture to", out_dir, "\n")
