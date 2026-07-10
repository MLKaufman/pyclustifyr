#!/usr/bin/env Rscript
# Exports clustifyr's bundled .rda datasets to portable formats
# (CSV / Matrix Market) under tests/data/ for use in pytest parity tests.
#
# The R source used to be vendored in this repo as a git submodule at
# `clustifyr/`; it has since been removed since tests/data/ already has
# everything committed. To regenerate these fixtures against a newer
# upstream release, clone https://github.com/rnabioco/clustifyr somewhere
# and point CLUSTIFYR_R_SRC at it, e.g.:
#   git clone https://github.com/rnabioco/clustifyr /tmp/clustifyr
#   CLUSTIFYR_R_SRC=/tmp/clustifyr Rscript scripts/export_r_datasets.R

suppressMessages(library(Matrix))

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
repo_root <- normalizePath(file.path(dirname(script_path), ".."))
r_src <- Sys.getenv("CLUSTIFYR_R_SRC", unset = file.path(repo_root, "clustifyr"))
if (!dir.exists(r_src)) {
  stop(
    "clustifyr R source not found at '", r_src, "'.\n",
    "Clone https://github.com/rnabioco/clustifyr and set CLUSTIFYR_R_SRC ",
    "to point at it (see comment at top of this script)."
  )
}
data_dir <- file.path(r_src, "data")
out_dir <- file.path(repo_root, "tests", "data")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

load_one <- function(name) {
  e <- new.env()
  load(file.path(data_dir, paste0(name, ".rda")), envir = e)
  get(name, envir = e)
}

write_vector <- function(x, name) {
  writeLines(as.character(x), file.path(out_dir, paste0(name, ".txt")))
}

write_df <- function(df, name, row_names = TRUE) {
  write.csv(df, file.path(out_dir, paste0(name, ".csv")), row.names = row_names)
}

write_dense_matrix <- function(mat, name) {
  df <- as.data.frame(as.matrix(mat), check.names = FALSE)
  write.csv(df, file.path(out_dir, paste0(name, ".csv")), row.names = TRUE)
}

write_sparse_matrix <- function(mat, name) {
  mat <- as(mat, "CsparseMatrix")
  mtx_path <- file.path(out_dir, paste0(name, ".mtx"))
  Matrix::writeMM(mat, mtx_path)
  system2("gzip", c("-f", mtx_path))
  writeLines(rownames(mat), file.path(out_dir, paste0(name, "_rownames.txt")))
  writeLines(colnames(mat), file.path(out_dir, paste0(name, "_colnames.txt")))
}

# --- matrices ---
write_sparse_matrix(load_one("pbmc_matrix_small"), "pbmc_matrix_small")
write_dense_matrix(load_one("cbmc_ref"), "cbmc_ref")

# --- data frames ---
write_df(load_one("cbmc_m"), "cbmc_m")
write_df(load_one("pbmc_meta"), "pbmc_meta")
write_df(load_one("pbmc_markers"), "pbmc_markers", row_names = FALSE)
write_df(load_one("pbmc_markers_M3Drop"), "pbmc_markers_M3Drop", row_names = FALSE)
write_df(load_one("downrefs"), "downrefs", row_names = FALSE)
write_df(load_one("object_loc_lookup"), "object_loc_lookup", row_names = FALSE)

# --- vectors ---
write_vector(load_one("pbmc_vargenes"), "pbmc_vargenes")
write_vector(load_one("human_genes_10x"), "human_genes_10x")
write_vector(load_one("mouse_genes_10x"), "mouse_genes_10x")

cat("Exported datasets to", out_dir, "\n")
