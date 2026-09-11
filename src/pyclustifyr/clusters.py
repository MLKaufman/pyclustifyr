"""Cluster-level aggregation utilities.

Python port of clustifyr's R/utils.R (average_clusters, percent_clusters,
overcluster, get_best_match_matrix, get_best_str, assign_ident).
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata, trim_mean


def _cluster_ids_from_metadata(mat: pd.DataFrame, metadata, cluster_col: str | None) -> list:
    if isinstance(metadata, pd.DataFrame):
        if cluster_col is None or cluster_col not in metadata.columns:
            raise ValueError("given `cluster_col` is not a column in `metadata`")
        cluster_info = metadata[cluster_col]
        if isinstance(cluster_info.dtype, pd.CategoricalDtype):
            cluster_info = cluster_info.cat.remove_unused_categories()
        cluster_ids = list(cluster_info)
    else:
        cluster_ids = list(metadata)
    if mat.shape[1] != len(cluster_ids):
        raise ValueError(
            "cluster assignments do not match the number of columns in the matrix"
        )
    return cluster_ids


def overcluster(mat: pd.DataFrame, cluster_id: dict[str, list], power: float = 0.15) -> dict[str, list]:
    """Sub-cluster each group with k-means, using ``n_cells**power`` centers."""
    from scipy.cluster.vq import kmeans2

    new_ids: dict[str, list] = {}
    for name, ids in cluster_id.items():
        if len(ids) > 1:
            sub = mat[ids].to_numpy(dtype=float).T
            k = max(1, int(len(ids) ** power))
            _, labels = kmeans2(sub, k, minit="++", seed=0)
            grouped: dict[int, list] = defaultdict(list)
            for cell_id, label in zip(ids, labels):
                grouped[int(label)].append(cell_id)
            for label, cell_ids in grouped.items():
                new_ids[f"{name}_{label}"] = cell_ids
        else:
            new_ids[name] = ids
    return new_ids


def average_clusters(
    mat: pd.DataFrame,
    metadata,
    cluster_col: str = "cluster",
    if_log: bool = True,
    cell_col: str | None = None,
    low_threshold: int = 0,
    method: str = "mean",
    output_log: bool = True,
    subclusterpower: float = 0,
    cut_n: int | None = None,
) -> pd.DataFrame:
    """Average (or median/trimean/etc.) expression per cluster.

    Parameters mirror clustifyr's ``average_clusters()``. ``mat`` is a genes x
    cells expression matrix; ``metadata`` is either a vector of per-cell
    cluster assignments (aligned positionally with ``mat``'s columns) or a
    DataFrame with ``cluster_col``.
    """
    if cell_col is not None and isinstance(metadata, pd.DataFrame):
        wanted = list(metadata[cell_col])
        if list(mat.columns) != wanted:
            mat = mat[wanted]

    cluster_ids = _cluster_ids_from_metadata(mat, metadata, cluster_col)

    # Mirrors R's split(): cells with a missing (NA) cluster label are dropped
    # entirely rather than forming their own group.
    groups: dict[object, list] = defaultdict(list)
    for col, cid in zip(mat.columns, cluster_ids):
        if pd.isna(cid):
            continue
        groups[cid].append(col)

    # Mirror R's split(): a plain (non-factor) grouping vector is coerced via
    # as.factor(), which sorts unique values; an explicit pandas Categorical
    # (R factor) keeps its own category order instead.
    if (
        isinstance(metadata, pd.DataFrame)
        and isinstance(metadata[cluster_col].dtype, pd.CategoricalDtype)
    ):
        ordered_keys = [c for c in metadata[cluster_col].cat.categories if c in groups]
    else:
        try:
            ordered_keys = sorted(groups.keys())
        except TypeError:
            # Missing-label sentinels may coexist with numeric cluster IDs.
            ordered_keys = sorted(groups.keys(), key=lambda value: (type(value).__name__, str(value)))
    groups = {k: groups[k] for k in ordered_keys}

    if subclusterpower > 0:
        groups = overcluster(mat, dict(groups), power=subclusterpower)

    def _agg(cell_ids: list) -> np.ndarray:
        sub = mat[cell_ids].to_numpy(dtype=float)
        if method == "mean":
            data = np.expm1(sub) if if_log else sub
            res = np.nanmean(data, axis=1)
            if output_log:
                res = np.log1p(res)
        elif method == "median":
            res = np.nanmedian(sub, axis=1)
            res = np.nan_to_num(res, nan=0.0)
        elif method == "trimean":
            q1 = np.nanpercentile(sub, 25, axis=1)
            q2 = np.nanpercentile(sub, 50, axis=1)
            q3 = np.nanpercentile(sub, 75, axis=1)
            res = 0.5 * q2 + 0.25 * q1 + 0.25 * q3
            res = np.nan_to_num(res, nan=0.0)
        elif method == "truncate":
            res = trim_mean(sub, proportiontocut=0.1, axis=1)
        elif method == "min":
            res = np.nanmin(sub, axis=1)
            res = np.nan_to_num(res, nan=0.0)
        elif method == "max":
            res = np.nanmax(sub, axis=1)
            res = np.nan_to_num(res, nan=0.0)
        else:
            raise ValueError(f"unsupported method: {method}")
        return res

    cluster_names = list(groups.keys())
    out = pd.DataFrame(
        {name: _agg(ids) for name, ids in groups.items()},
        index=mat.index,
    )[cluster_names]

    counts = {name: len(ids) for name, ids in groups.items()}
    if low_threshold > 0:
        keep = [name for name in cluster_names if counts[name] >= low_threshold]
        dropped = [name for name in cluster_names if counts[name] < low_threshold]
        if dropped:
            warnings.warn(
                f"The following clusters have less than {low_threshold} cells for this "
                f"analysis: {', '.join(map(str, dropped))}. They are excluded."
            )
        out = out[keep]
    else:
        dropped = [name for name in cluster_names if counts[name] < 10]
        if dropped:
            warnings.warn(
                f"The following clusters have less than 10 cells for this analysis: "
                f"{', '.join(map(str, dropped))}. Classification is likely inaccurate."
            )

    if cut_n is not None:
        arr = out.to_numpy()
        ranks = np.apply_along_axis(lambda col: rankdata(-col, method="average"), axis=0, arr=arr)
        arr = np.where(ranks > cut_n, 0, arr)
        out = pd.DataFrame(arr, index=out.index, columns=out.columns)

    return out


def percent_clusters(
    mat: pd.DataFrame,
    metadata,
    cluster_col: str = "cluster",
    cut_num: float = 0.5,
) -> pd.DataFrame:
    """Fraction of cells per cluster with expression above ``cut_num``."""
    binarized = mat.copy()
    binarized = binarized.where(binarized < cut_num, 1)
    binarized = binarized.where(binarized >= cut_num, 0)
    return average_clusters(binarized, metadata, cluster_col=cluster_col, if_log=False)


def get_best_match_matrix(cor_mat: pd.DataFrame) -> pd.DataFrame:
    """Binarize a correlation matrix, marking the row-wise best match(es) as 1."""
    row_max = cor_mat.max(axis=1)
    return (cor_mat.sub(row_max, axis=0) == 0).astype(int)


def get_best_str(
    name,
    best_mat: pd.DataFrame,
    cor_mat: pd.DataFrame,
    carry_cor: bool = True,
) -> str:
    """Render the best call(s) for a row as a display string, e.g. ``B (0.82)``."""
    row = best_mat.loc[name]
    if row.sum() == 0:
        return "?"
    best_names = row.index[row == 1]
    parts = []
    for col in best_names:
        r = round(float(cor_mat.loc[name, col]), 2)
        parts.append(f"{col} ({r})" if carry_cor else f"{col}")
    return "; ".join(parts)


def assign_ident(
    metadata: pd.DataFrame,
    clusters,
    idents,
    cluster_col: str = "cluster",
    ident_col: str = "type",
) -> pd.DataFrame:
    """Manually overwrite ``ident_col`` for the given ``clusters``."""
    clusters = list(clusters) if not isinstance(clusters, str) else [clusters]
    if isinstance(idents, str):
        idents = [idents]
    else:
        idents = list(idents)
    if len(idents) == 1:
        idents = idents * len(clusters)
    elif len(idents) != len(clusters):
        raise ValueError("unsupported lengths pairs of clusters and idents")

    metadata = metadata.copy()
    for cluster, ident in zip(clusters, idents):
        metadata.loc[metadata[cluster_col] == cluster, ident_col] = ident
    return metadata
