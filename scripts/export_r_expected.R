#!/usr/bin/env Rscript
# Computes expected outputs from the installed clustifyr R package for use as
# pytest parity fixtures under tests/data/expected/.

suppressMessages(library(clustifyr))
suppressMessages(library(Matrix))

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
repo_root <- normalizePath(file.path(dirname(script_path), ".."))
out_dir <- file.path(repo_root, "tests", "data", "expected")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data(pbmc_matrix_small, envir = environment())
data(pbmc_meta, envir = environment())
data(cbmc_ref, envir = environment())
data(cbmc_m, envir = environment())
data(pbmc_vargenes, envir = environment())

save_csv <- function(x, name, row_names = TRUE) {
  write.csv(as.data.frame(as.matrix(x), check.names = FALSE), file.path(out_dir, paste0(name, ".csv")), row.names = row_names)
}

# --- average_clusters, all pseudobulk methods ---
for (m in c("mean", "median", "trimean", "truncate", "min", "max")) {
  res <- average_clusters(pbmc_matrix_small, pbmc_meta, cluster_col = "classified", if_log = FALSE, method = m)
  save_csv(res, paste0("avg_", m))
}
res_log <- average_clusters(pbmc_matrix_small, pbmc_meta, cluster_col = "classified", if_log = TRUE)
save_csv(res_log, "avg_mean_iflog")

# --- clustify(), across compute methods ---
for (cm in c("spearman", "pearson", "cosine", "kendall")) {
  res <- clustify(
    input = pbmc_matrix_small,
    metadata = pbmc_meta,
    ref_mat = cbmc_ref,
    cluster_col = "classified",
    compute_method = cm,
    verbose = FALSE
  )
  save_csv(res, paste0("clustify_", cm))
}

# per_cell clustify (subset of cells for speed)
small_cells <- colnames(pbmc_matrix_small)[1:50]
res_percell <- clustify(
  input = pbmc_matrix_small[, small_cells],
  metadata = pbmc_meta[small_cells, ],
  ref_mat = cbmc_ref,
  cluster_col = "classified",
  per_cell = TRUE,
  verbose = FALSE
)
save_csv(res_percell, "clustify_percell")

# --- cor_to_call / cor_to_call_rank / cor_to_call_topn ---
res <- clustify(
  input = pbmc_matrix_small,
  metadata = pbmc_meta,
  ref_mat = cbmc_ref,
  cluster_col = "classified",
  verbose = FALSE
)
write.csv(cor_to_call(res), file.path(out_dir, "cor_to_call.csv"), row.names = FALSE)
write.csv(cor_to_call(res, carry_r = TRUE, threshold = 0.85), file.path(out_dir, "cor_to_call_threshold.csv"), row.names = FALSE)
write.csv(cor_to_call_rank(res, threshold = "auto"), file.path(out_dir, "cor_to_call_rank.csv"), row.names = FALSE)
write.csv(cor_to_call_topn(res, threshold = 0.5, topn = 3), file.path(out_dir, "cor_to_call_topn.csv"), row.names = FALSE)

# --- vec_out / call_to_metadata ---
vec_res <- clustify(
  input = pbmc_matrix_small,
  metadata = pbmc_meta,
  ref_mat = cbmc_ref,
  cluster_col = "classified",
  vec_out = TRUE,
  verbose = FALSE
)
writeLines(as.character(vec_res), file.path(out_dir, "clustify_vec_out.txt"))

# --- compare_genelist.R ---
data(pbmc_markers, envir = environment())
pbmc_avg <- average_clusters(pbmc_matrix_small, pbmc_meta, cluster_col = "classified")
pbmc_avgb <- binarize_expr(pbmc_avg)
save_csv(pbmc_avgb, "pbmc_avgb")

mm <- matrixize_markers(pbmc_markers)
save_csv(mm, "mm_unranked")
mm_ranked <- matrixize_markers(pbmc_markers, ranked = TRUE)
save_csv(mm_ranked, "mm_ranked")

for (metric in c("hyper", "jaccard", "spearman")) {
  res_cl <- compare_lists(pbmc_avgb, mm, metric = metric)
  save_csv(res_cl, paste0("compare_", metric))
}
res_cbmc <- compare_lists(pbmc_avgb, cbmc_m, metric = "hyper")
save_csv(res_cbmc, "compare_hyper_cbmc")

# --- markers.py ---
res_rms <- ref_marker_select(cbmc_ref, cut = 2)
write.csv(res_rms, file.path(out_dir, "ref_marker_select.csv"), row.names = FALSE)

m1 <- pos_neg_marker(cbmc_m)
save_csv(m1, "pos_neg_marker")

m2 <- reverse_marker_matrix(cbmc_m)
write.csv(m2, file.path(out_dir, "reverse_marker_matrix.csv"), row.names = FALSE)

pn_ref <- data.frame("Myeloid" = c(1, 0.01, 0), row.names = c("CD74", "clustifyr0", "CD79A"))
res_pn <- pos_neg_select(
  input = pbmc_matrix_small,
  ref_mat = pn_ref,
  metadata = pbmc_meta,
  cluster_col = "classified",
  cutoff_score = 0.8
)
save_csv(res_pn, "pos_neg_select")

gpm <- gene_pct_markerm(pbmc_matrix_small, cbmc_m, pbmc_meta, cluster_col = "classified")
save_csv(gpm, "gene_pct_markerm")

fpca <- feature_select_PCA(cbmc_ref, if_log = FALSE)
writeLines(fpca, file.path(out_dir, "feature_select_pca.txt"))

rfs <- ref_feature_select(cbmc_ref, n = 50, mode = "var")
writeLines(rfs, file.path(out_dir, "ref_feature_select.txt"))

comb <- make_comb_ref(cbmc_ref, sep = "_+_")
save_csv(comb, "make_comb_ref")

ag <- append_genes(c("PPBP", "NOTAGENE123"), cbmc_ref)
save_csv(ag, "append_genes")

set.seed(1)
cbs <- paste0("cb_", 1:100)
spatial_coords <- data.frame(row.names = cbs, X = runif(100), Y = runif(100))
group_ids <- sample(c("A", "B"), 100, replace = TRUE)
dist_res <- calc_distance(spatial_coords, group_ids)
save_csv(dist_res, "calc_distance")
write.csv(spatial_coords, file.path(out_dir, "spatial_coords.csv"), row.names = TRUE)
writeLines(group_ids, file.path(out_dir, "group_ids.txt"))

for (metric in c("hyper", "jaccard", "pct")) {
  res_cll <- clustify_lists(
    input = pbmc_matrix_small,
    marker = cbmc_m,
    metadata = pbmc_meta,
    cluster_col = "classified",
    verbose = FALSE,
    metric = metric
  )
  save_csv(res_cll, paste0("clustify_lists_", metric))
}

# --- clustify.Seurat / object_access ---
suppressMessages(library(SeuratObject))
so <- so_pbmc()
res_so <- clustify(so, cbmc_ref, cluster_col = "seurat_clusters", obj_out = FALSE, verbose = FALSE)
save_csv(res_so, "clustify_seurat_cormat")

so2 <- clustify(so, cbmc_ref, cluster_col = "seurat_clusters", obj_out = TRUE, verbose = FALSE)
write.csv(so2@meta.data[, c("seurat_clusters", "type", "r")], file.path(out_dir, "clustify_seurat_obj_meta.csv"), row.names = TRUE)

cat("Exported expected outputs to", out_dir, "\n")
