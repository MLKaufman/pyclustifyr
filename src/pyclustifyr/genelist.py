"""Compare scRNA-seq expression to marker gene lists.

Python port of clustifyr's R/compare_genelist.R.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, rankdata

_RP_PATTERN = re.compile(r"^RP[0-9,LS]|^Rp[0-9,ls]")


def binarize_expr(mat: pd.DataFrame, n: int = 1000, cut: float = 0) -> pd.DataFrame:
    """Keep the top ``n`` genes per column with expression strictly above ``cut``."""
    arr = mat.to_numpy(dtype=float)
    selected = arr > cut
    if n < mat.shape[0]:
        ranks = np.apply_along_axis(lambda col: rankdata(-col, method="average"), axis=0, arr=arr)
        selected &= ranks <= n
    out = selected.astype(int)
    return pd.DataFrame(out, index=mat.index, columns=mat.columns)


def _cluster_order(df: pd.DataFrame, cluster_col: str) -> list:
    col = df[cluster_col]
    if isinstance(col.dtype, pd.CategoricalDtype):
        present = set(col)
        return [c for c in col.cat.categories if c in present]
    return sorted(col.unique())


def matrixize_markers(
    marker_df: pd.DataFrame,
    ranked: bool = False,
    n: int | None = None,
    step_weight: float = 1,
    background_weight: float = 0,
    unique: bool = False,
    remove_rp: bool = False,
) -> pd.DataFrame:
    """Convert a long-format marker table into a wide gene x cluster matrix.

    If ``marker_df`` has no ``gene``/``cluster`` columns, it is treated as a
    wide matrix of marker genes (one column per cluster) and reshaped to long
    form first, preserving column order as the cluster ordering.
    """
    df = marker_df.copy()

    if "feature" in df.columns and "group" in df.columns:
        df = df.rename(columns={"feature": "gene", "group": "cluster"})
        df = df.sort_values(["cluster", "padj"], kind="stable")

    if "gene" not in df.columns:
        cluster_order = list(df.columns)
        df = df.melt(var_name="cluster", value_name="gene")
        df["cluster"] = pd.Categorical(df["cluster"], categories=cluster_order)
    else:
        cluster_order = _cluster_order(df, "cluster")

    df = df.dropna(subset=["gene", "cluster"])
    df["gene"] = df["gene"].astype(str)

    if remove_rp:
        df = df[~df["gene"].str.contains(_RP_PATTERN)]

    if unique:
        counts = df.groupby("gene")["gene"].transform("size")
        df = df[counts == 1]

    if df.empty:
        return pd.DataFrame(columns=cluster_order, dtype=float if ranked else object)
    cut_num = df.groupby("cluster", observed=True).size().min()
    if n is not None and n < cut_num:
        cut_num = n

    marker = df[["gene", "cluster"]].groupby("cluster", observed=True, group_keys=False).head(cut_num)
    marker = marker.copy()
    marker["pos"] = marker.groupby("cluster", observed=True).cumcount()

    if ranked:
        marker["weight"] = step_weight * (cut_num - marker["pos"]) + background_weight
        wide = marker.pivot(index="gene", columns="cluster", values="weight")
        wide = wide.reindex(columns=cluster_order).fillna(0)
    else:
        marker["pos"] = marker["pos"] + 1
        wide = marker.pivot(index="pos", columns="cluster", values="gene")
        wide = wide.reindex(columns=cluster_order)
        wide.index.name = None

    wide.columns.name = None
    return wide


def get_vargenes(marker_mat: pd.DataFrame) -> list:
    """Flatten a marker matrix (from ``matrixize_markers``) into a gene list."""
    if str(marker_mat.index[0]) != "1":
        return list(pd.unique(marker_mat.index))
    return list(pd.unique(marker_mat.to_numpy().ravel()))


def _marker_col(marker_mat: pd.DataFrame, col) -> list:
    return [g for g in marker_mat[col].to_numpy().ravel() if pd.notna(g)]


def compare_lists(
    bin_mat: pd.DataFrame,
    marker_mat: pd.DataFrame,
    n: int = 30000,
    metric: str = "hyper",
    output_high: bool = True,
    details_out: bool = False,
):
    """Score how well each column of ``bin_mat`` overlaps with each marker set.

    Hypergeometric scoring requires a positive integer universe size large
    enough for the compared sets. Rank-distance ("spearman") comparisons
    with fewer than two shared genes return NaN rather than a perfect match.
    """
    if metric not in {"hyper", "jaccard", "spearman", "rank_distance", "gsea"}:
        raise ValueError(f"Unknown metric: {metric}")
    if metric == "rank_distance":
        metric = "spearman"
    if metric in {"hyper", "jaccard"} and not bin_mat.isin([0, 1]).all().all():
        raise ValueError("hyper/jaccard require binary input (0/1); use rank_distance for ranked input")

    marker_cols = list(marker_mat.columns)
    bin_cols = list(bin_mat.columns)

    details = None
    if details_out:
        details = pd.DataFrame(index=bin_cols, columns=marker_cols, dtype=object)
        for x in bin_cols:
            list_top = list(bin_mat.index[bin_mat[x] == 1])
            for y in marker_cols:
                marker_list = _marker_col(marker_mat, y)
                details.loc[x, y] = ",".join(sorted(set(list_top) & set(marker_list), key=list_top.index))

    if metric == "hyper":
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("gene universe size n must be a positive integer")
        out = np.empty((len(bin_cols), len(marker_cols)))
        for i, x in enumerate(bin_cols):
            list_top = list(bin_mat.index[bin_mat[x] == 1])
            raw = []
            for y in marker_cols:
                marker_list = _marker_col(marker_mat, y)
                t = len(set(list_top) & set(marker_list))
                a = max(len(list_top), len(marker_list))
                b = min(len(list_top), len(marker_list))
                if n < a:
                    raise ValueError("gene universe size n is smaller than a query or marker set")
                pval = sum(hypergeom.pmf(k, n, a, b) for k in range(t, b + 1))
                raw.append(pval)
            out[i] = _p_adjust_holm(raw)
        res = pd.DataFrame(out, index=bin_cols, columns=marker_cols)

    elif metric == "jaccard":
        out = np.empty((len(bin_cols), len(marker_cols)))
        for i, x in enumerate(bin_cols):
            list_top = set(bin_mat.index[bin_mat[x] == 1])
            for j, y in enumerate(marker_cols):
                marker_list = set(_marker_col(marker_mat, y))
                inter = len(list_top & marker_list)
                out[i, j] = inter / (len(list_top) + len(marker_list) - inter)
        res = pd.DataFrame(out, index=bin_cols, columns=marker_cols)

    elif metric == "spearman":
        out = np.empty((len(bin_cols), len(marker_cols)))
        for i, x in enumerate(bin_cols):
            # kind="stable" matches R's order(): ties keep their original row order.
            bin_temp = bin_mat[x].sort_values(ascending=False, kind="stable")
            list_top = list(bin_temp.index)
            top_rank = {gene: rank for rank, gene in enumerate(list_top)}
            for j, y in enumerate(marker_cols):
                marker_list = _marker_col(marker_mat, y)
                v1 = [g for g in marker_list if g in top_rank]
                if len(set(v1)) < 2:
                    out[i, j] = np.nan
                    continue
                v2 = [g for g in list_top if g in v1]
                v2_rank = {gene: rank for rank, gene in enumerate(v2)}
                out[i, j] = sum(abs(i2 - v2_rank[g]) for i2, g in enumerate(v1))
        res = pd.DataFrame(out, index=bin_cols, columns=marker_cols)

    elif metric == "gsea":
        from .gsea import run_gsea

        pval_cols = {}
        for y in marker_cols:
            gsea_res = run_gsea(bin_mat, {y: _marker_col(marker_mat, y)}, n_perm=1000, per_cell=True)
            pval_cols[y] = gsea_res["pval"]
        res = pd.DataFrame(pval_cols)
    else:
        raise ValueError(f"unrecognized metric: {metric}")

    if output_high:
        if metric in ("hyper", "gsea"):
            res = -np.log10(res)
        elif metric == "spearman":
            res = -res + res.max().max()

    if details_out:
        return {"res": res, "details": details}
    return res


def _p_adjust_holm(pvals: list[float]) -> np.ndarray:
    """Holm step-down adjustment, matching R's default ``stats::p.adjust``."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.full(n, np.nan)
    running_max = 0.0
    for i, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            continue
        val = (n - i) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted
