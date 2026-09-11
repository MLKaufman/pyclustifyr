"""AnnData integration layer.

Python analog of clustifyr's R/object_access.R (which dispatches on Seurat /
SingleCellExperiment objects). AnnData is the natural Python equivalent of
those container types, so this module provides the same access pattern:
pull an expression matrix + metadata out of the object, run ``clustify``/
``clustify_lists``, and write the result back onto the object.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from .classify import call_to_metadata, cor_to_call
from .clusters import average_clusters
from .clustify import _marker_calls, _marker_per_cell, clustify, clustify_lists


def object_data(
    adata: ad.AnnData,
    slot: str = "data",
    layer: str | None = None,
    n_genes: int = 1000,
) -> pd.DataFrame | pd.DataFrame | list:
    """Pull expression data, metadata, or variable genes out of an AnnData object.

    ``slot`` mirrors clustifyr's ``object_data()``:

    - ``"data"``: genes x cells expression matrix (``adata.X``, or ``layer`` if given)
    - ``"meta.data"``: ``adata.obs``
    - ``"var.genes"``: highly-variable gene names (from ``adata.var["highly_variable"]``),
      truncated to ``n_genes`` (set to 0 to keep all)
    """
    if slot in ("data", "counts") or layer is not None:
        arr = adata.layers[layer] if layer is not None else adata.X
        arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
        return pd.DataFrame(arr.T, index=adata.var_names, columns=adata.obs_names)
    if slot == "meta.data":
        return adata.obs
    if slot == "var.genes":
        if "highly_variable" not in adata.var.columns:
            return []
        genes = list(adata.var_names[adata.var["highly_variable"]])
        if n_genes > 0 and len(genes) > n_genes:
            genes = genes[:n_genes]
        return genes
    raise ValueError(f"{slot} access method not implemented")


def write_meta(adata: ad.AnnData, meta: pd.DataFrame) -> ad.AnnData:
    """Return a copy with metadata aligned by unique, matching observation IDs."""
    if not adata.obs_names.is_unique or not meta.index.is_unique:
        raise ValueError("metadata and AnnData cell IDs must be unique")
    if (len(meta) != adata.n_obs or not adata.obs_names.isin(meta.index).all()):
        raise ValueError("metadata cell IDs must match AnnData observation IDs")
    out = adata.copy()
    out.obs = meta.reindex(adata.obs_names).copy(deep=True)
    return out


def object_ref(
    adata: ad.AnnData,
    cluster_col: str | None = None,
    var_genes_only: bool = False,
    method: str = "mean",
    if_log: bool = True,
) -> pd.DataFrame:
    """Build a reference (average expression per cluster) matrix from an AnnData object."""
    expr = object_data(adata, "data")
    metadata = object_data(adata, "meta.data")
    if var_genes_only:
        expr = expr.loc[object_data(adata, "var.genes", n_genes=0)]
    return average_clusters(expr, metadata, cluster_col=cluster_col, method=method, if_log=if_log)


def clustify_adata(
    adata: ad.AnnData,
    ref_mat: pd.DataFrame,
    cluster_col: str | None = None,
    query_genes: list | None = None,
    n_genes: int = 1000,
    use_var_genes: bool = True,
    layer: str | None = None,
    obj_out: bool = True,
    vec_out: bool = False,
    rename_prefix: str | None = None,
    threshold: float | str = "auto",
    per_cell: bool = False,
    **kwargs,
) -> ad.AnnData | pd.DataFrame | list | dict[str, pd.DataFrame]:
    """Classify an AnnData object's clusters (or cells) against a reference matrix.

    By default (``obj_out=True``), returns a copy of ``adata`` with the call
    (and correlation) written into ``.obs``. Set ``obj_out=False`` to get the
    raw similarity matrix instead, matching ``clustify(..., vec_out=False)``
    on a plain matrix. ``return_pvalues=True`` also requires ``obj_out=False``
    and ``vec_out=False``, returning the score/p_val dictionary from clustify.
    """
    if kwargs.get("return_pvalues", False) and (obj_out or vec_out):
        raise ValueError("return_pvalues requires obj_out=False and vec_out=False")
    expr = object_data(adata, "data", layer=layer)
    metadata = object_data(adata, "meta.data")

    if query_genes is None and use_var_genes:
        var_genes = object_data(adata, "var.genes", n_genes=n_genes)
        query_genes = var_genes or None

    res = clustify(
        expr,
        ref_mat,
        metadata=metadata,
        cluster_col=cluster_col,
        query_genes=query_genes,
        per_cell=per_cell,
        vec_out=vec_out,
        rename_prefix=rename_prefix,
        threshold=threshold,
        **kwargs,
    )

    if vec_out or not obj_out:
        return res

    df_temp = cor_to_call(res, metadata=metadata, cluster_col=cluster_col, threshold=threshold)
    df_full = call_to_metadata(
        df_temp, metadata=metadata, cluster_col=cluster_col, per_cell=per_cell, rename_prefix=rename_prefix
    )
    return write_meta(adata, df_full)


def clustify_lists_adata(
    adata: ad.AnnData,
    marker: pd.DataFrame,
    cluster_col: str | None = None,
    layer: str | None = None,
    obj_out: bool = True,
    vec_out: bool = False,
    rename_prefix: str | None = None,
    threshold: float = 0,
    per_cell: bool = False,
    **kwargs,
) -> ad.AnnData | pd.DataFrame | list:
    """``clustify_lists`` for an AnnData object; see ``clustify_adata`` for the ``obj_out`` contract."""
    expr = object_data(adata, "data", layer=layer)
    metadata = object_data(adata, "meta.data")

    res = clustify_lists(
        expr,
        marker,
        metadata=metadata,
        cluster_col=cluster_col,
        per_cell=per_cell,
        vec_out=vec_out,
        rename_prefix=rename_prefix,
        threshold=threshold,
        **kwargs,
    )

    if vec_out or not obj_out:
        return res

    metric = kwargs.get("metric", "hyper")
    scores = res["res"] if isinstance(res, dict) else res
    df_temp = _marker_calls(scores, metric, kwargs.get("output_high", True), cluster_col, threshold)
    output_per_cell = _marker_per_cell(metric, per_cell, kwargs.get("input_markers", False))
    df_full = call_to_metadata(
        df_temp, metadata=metadata, cluster_col=cluster_col,
        per_cell=output_per_cell, rename_prefix=rename_prefix,
    )
    return write_meta(adata, df_full)
