"""pyclustifyr: a Python port of the R package clustifyr.

Classify single-cell RNA-seq clusters (or individual cells) against reference
expression data or marker gene lists.
"""

from .anndata_io import clustify_adata, clustify_lists_adata, object_data, object_ref, write_meta
from .cellbrowsers import get_ucsc_reference
from .classify import (
    call_consensus,
    call_to_metadata,
    collapse_to_cluster,
    cor_to_call,
    cor_to_call_rank,
    cor_to_call_topn,
)
from .clusters import (
    assign_ident,
    average_clusters,
    get_best_match_matrix,
    get_best_str,
    overcluster,
    percent_clusters,
)
from .clustify import clustify, clustify_lists
from .genelist import binarize_expr, compare_lists, get_vargenes, matrixize_markers
from .gsea import calc_gsea_stat, calculate_pathway_gsea, fgsea_simple, gmt_to_list, plot_pathway_gsea, run_gsea
from .markers import (
    append_genes,
    calc_distance,
    check_raw_counts,
    downsample_matrix,
    feature_select_pca,
    gene_pct,
    gene_pct_markerm,
    make_comb_ref,
    marker_select,
    pos_neg_marker,
    pos_neg_select,
    ref_feature_select,
    ref_marker_select,
    reverse_marker_matrix,
)
from .plot import plot_best_call, plot_cor, plot_cor_heatmap, plot_dims, plot_gene
from .similarity import calc_similarity, clustifyr_methods, cosine, get_similarity, kl_divergence

__all__ = [
    "append_genes",
    "assign_ident",
    "average_clusters",
    "binarize_expr",
    "calc_distance",
    "calc_gsea_stat",
    "calc_similarity",
    "calculate_pathway_gsea",
    "call_consensus",
    "call_to_metadata",
    "check_raw_counts",
    "clustify",
    "clustify_adata",
    "clustify_lists",
    "clustify_lists_adata",
    "clustifyr_methods",
    "collapse_to_cluster",
    "compare_lists",
    "cor_to_call",
    "cor_to_call_rank",
    "cor_to_call_topn",
    "cosine",
    "downsample_matrix",
    "feature_select_pca",
    "fgsea_simple",
    "gene_pct",
    "gene_pct_markerm",
    "get_best_match_matrix",
    "get_best_str",
    "get_similarity",
    "get_ucsc_reference",
    "get_vargenes",
    "gmt_to_list",
    "kl_divergence",
    "make_comb_ref",
    "marker_select",
    "matrixize_markers",
    "object_data",
    "object_ref",
    "overcluster",
    "percent_clusters",
    "plot_best_call",
    "plot_cor",
    "plot_cor_heatmap",
    "plot_dims",
    "plot_gene",
    "plot_pathway_gsea",
    "pos_neg_marker",
    "pos_neg_select",
    "ref_feature_select",
    "ref_marker_select",
    "reverse_marker_matrix",
    "run_gsea",
    "write_meta",
]
