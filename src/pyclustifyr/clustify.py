"""Main entry point: compare scRNA-seq data to reference data.

Python port of clustifyr's R/main.R (``clustify.default``).
"""

from __future__ import annotations

import pandas as pd

from .classify import call_consensus, call_to_metadata, cor_to_call, cor_to_call_rank
from .clusters import average_clusters
from .genelist import binarize_expr, compare_lists, matrixize_markers
from .markers import gene_pct_markerm, pos_neg_marker, pos_neg_select
from .similarity import clustifyr_methods, get_similarity, permute_similarity


def _common_elements(*vecs) -> list:
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return []
    result = list(vecs[0])
    for v in vecs[1:]:
        v_set = set(v)
        result = [x for x in result if x in v_set]
    return result


def clustify(
    input: pd.DataFrame,
    ref_mat: pd.DataFrame,
    metadata: pd.DataFrame | pd.Series | list | None = None,
    cluster_col: str | None = None,
    query_genes: list | None = None,
    per_cell: bool = False,
    n_perm: int = 0,
    compute_method: str = "spearman",
    pseudobulk_method: str = "mean",
    verbose: bool = True,
    rm0: bool = False,
    vec_out: bool = False,
    rename_prefix: str | None = None,
    threshold: float | str = "auto",
    low_threshold_cell: int = 0,
    exclude_genes: list | None = None,
    if_log: bool = True,
    rng=None,
    **kwargs,
) -> pd.DataFrame | list:
    """Classify clusters (or single cells) by similarity to a reference matrix.

    ``input`` is a genes x cells expression matrix. ``metadata`` supplies
    per-cell cluster assignments (a vector, or a DataFrame plus
    ``cluster_col``) and is required unless ``per_cell=True``.

    Returns a clusters x cell-types similarity matrix by default, or a list of
    per-cluster/per-cell type calls if ``vec_out=True``.
    """
    if compute_method not in clustifyr_methods:
        raise ValueError(f"{compute_method} correlation method not implemented")

    if metadata is None and not per_cell:
        raise ValueError("`metadata` needed for per cluster analysis")

    if cluster_col is not None and isinstance(metadata, pd.DataFrame) and cluster_col not in metadata.columns:
        raise ValueError("given `cluster_col` is not a column in `metadata`")

    exclude_genes = exclude_genes or []

    gene_constraints = _common_elements(input.index, ref_mat.index, query_genes)
    if exclude_genes:
        exclude_set = set(exclude_genes)
        gene_constraints = [g for g in gene_constraints if g not in exclude_set]

    if verbose:
        msg = f"using # of genes: {len(gene_constraints)}"
        if len(gene_constraints) >= 2000:
            msg += (
                "\nusing a high number (>=2000) genes to calculate correlation "
                "please consider feature selection to improve performance"
            )
        print(msg)

    expr_mat = input.loc[gene_constraints]
    ref_mat = ref_mat.loc[gene_constraints]

    if not per_cell:
        if isinstance(metadata, pd.DataFrame):
            if cluster_col is None:
                raise ValueError("cluster_col is required when metadata is a DataFrame")
            cluster_ids = list(metadata[cluster_col])
        else:
            cluster_ids = list(metadata)
        cluster_ids = ["orig.NA" if pd.isna(c) else c for c in cluster_ids]
    else:
        cluster_ids = list(expr_mat.columns)

    if n_perm == 0:
        res = get_similarity(
            expr_mat,
            ref_mat,
            cluster_ids=cluster_ids,
            per_cell=per_cell,
            compute_method=compute_method,
            pseudobulk_method=pseudobulk_method,
            rm0=rm0,
            if_log=if_log,
            low_threshold=low_threshold_cell,
            **kwargs,
        )
    else:
        result = permute_similarity(
            expr_mat,
            ref_mat,
            cluster_ids=cluster_ids,
            n_perm=n_perm,
            per_cell=per_cell,
            compute_method=compute_method,
            pseudobulk_method=pseudobulk_method,
            rm0=rm0,
            rng=rng,
            **kwargs,
        )
        res = result["score"]

    if verbose:
        print(f"similarity computation completed, matrix of {res.shape[0]} x {res.shape[1]}, preparing output")

    if not vec_out:
        return res

    df_temp = cor_to_call(res, metadata=metadata, cluster_col=cluster_col, threshold=threshold)
    df_temp_full = call_to_metadata(
        df_temp, metadata=metadata, cluster_col=cluster_col, per_cell=per_cell, rename_prefix=rename_prefix
    )
    col = f"{rename_prefix}_type" if rename_prefix else "type"
    return list(df_temp_full[col])


def clustify_lists(
    input: pd.DataFrame,
    marker: pd.DataFrame,
    marker_inmatrix: bool = True,
    metadata: pd.DataFrame | pd.Series | list | None = None,
    cluster_col: str | None = None,
    if_log: bool = True,
    per_cell: bool = False,
    topn: int = 800,
    cut: float = 0,
    genome_n: int = 30000,
    metric: str = "hyper",
    output_high: bool = True,
    vec_out: bool = False,
    rename_prefix: str | None = None,
    threshold: float = 0,
    low_threshold_cell: int = 0,
    verbose: bool = True,
    input_markers: bool = False,
    details_out: bool = False,
    **matrixize_kwargs,
) -> pd.DataFrame | list:
    """Classify clusters (or cells) by overlap with reference marker gene lists.

    ``metric`` selects the scoring method: ``"hyper"`` (hypergeometric
    enrichment, default), ``"jaccard"``, ``"spearman"``, ``"pct"`` (percent of
    cells expressing each marker set), ``"posneg"`` (positive/negative marker
    scoring), or ``"consensus"`` (combines hyper/jaccard/pct/posneg ranks).
    """
    if metric in ("posneg", "pct"):
        per_cell = True
    if input_markers:
        per_cell = True

    if not per_cell:
        input_avg = average_clusters(
            input, metadata, cluster_col=cluster_col, if_log=if_log, low_threshold=low_threshold_cell
        )
    else:
        input_avg = input

    if not input_markers:
        bin_input = binarize_expr(input_avg, n=topn, cut=cut)
    else:
        bin_input = input

    if not marker_inmatrix and metric != "posneg":
        marker = matrixize_markers(marker, **matrixize_kwargs)
        if verbose:
            print(f"number of total markers: {marker.shape[0]}")

    if metric == "consensus":
        results = [
            clustify_lists(input, marker, metadata=metadata, cluster_col=cluster_col, metric=m, verbose=False)
            for m in ("hyper", "jaccard", "pct", "posneg")
        ]
        call_list = [cor_to_call_rank(r, cluster_col=cluster_col or "cluster") for r in results]
        res = call_consensus(call_list)
    elif metric == "pct":
        res = gene_pct_markerm(input_avg, marker, metadata, cluster_col=cluster_col)
    elif metric == "gsea":
        res = compare_lists(bin_input, marker, n=genome_n, metric="gsea", output_high=output_high)
    elif metric != "posneg":
        res = compare_lists(
            bin_input, marker, n=genome_n, metric=metric, output_high=output_high, details_out=details_out
        )
    else:
        if not all(pd.api.types.is_numeric_dtype(marker[c]) for c in marker.columns):
            marker = pos_neg_marker(marker)
        res = pos_neg_select(input_avg, marker, metadata, cluster_col=cluster_col)

    if verbose:
        print(f"similarity computation completed, matrix of {res.shape[0]} x {res.shape[1]}, preparing output")

    if not vec_out:
        return res

    if metric != "consensus":
        df_temp = cor_to_call(res, metadata=metadata, cluster_col=cluster_col, threshold=threshold)
        df_temp_full = call_to_metadata(
            df_temp, metadata=metadata, cluster_col=cluster_col, per_cell=per_cell, rename_prefix=rename_prefix
        )
    else:
        df_temp_full = res

    col = f"{rename_prefix}_type" if rename_prefix else "type"
    return list(df_temp_full[col])
