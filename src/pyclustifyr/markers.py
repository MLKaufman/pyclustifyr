"""Marker selection, positive/negative marker scoring, and misc reference utilities.

Python port of clustifyr's R/utils.R marker-related functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .clusters import _cluster_ids_from_metadata, average_clusters


def marker_select(row: pd.Series, cut: float = 1, compto: int = 1) -> tuple | None:
    """Decide whether a single gene (row of cluster-averaged expression) is a marker.

    Returns ``(best_cluster, ratio)`` where ``ratio`` is the expression of the
    ``compto``-th runner-up cluster divided by the top cluster's expression
    (lower is more specific), or ``None`` if the top value is below ``cut``.
    """
    sorted_row = row.sort_values(ascending=False)
    if sorted_row.iloc[0] >= cut:
        return sorted_row.index[0], float(sorted_row.iloc[compto] / sorted_row.iloc[0])
    return None


def ref_marker_select(
    mat: pd.DataFrame,
    cut: float = 0.5,
    arrange: bool = True,
    compto: int = 1,
) -> pd.DataFrame:
    """Find, for every gene, the cluster it's most specific to."""
    mat = mat[mat.index.notna()]
    mat = mat[mat.sum(axis=1) != 0]

    rows = []
    for gene, row in mat.iterrows():
        result = marker_select(row, cut=cut, compto=compto)
        if result is not None:
            rows.append((gene, result[0], result[1]))

    res = pd.DataFrame(rows, columns=["gene", "cluster", "ratio"])
    if arrange:
        res = res.sort_values(["cluster", "ratio"], kind="stable").reset_index(drop=True)
    return res


def pos_neg_marker(mat: pd.DataFrame) -> pd.DataFrame:
    """Build a 0/1 gene x cell-type matrix from a wide table of positive markers."""
    long = mat.melt(var_name="type", value_name="gene").dropna(subset=["gene"])
    long["expression"] = 1
    wide = long.pivot_table(index="gene", columns="type", values="expression", aggfunc="first")
    wide = wide.reindex(columns=mat.columns).fillna(0)
    wide.columns.name = None
    wide.index.name = None
    return wide


def reverse_marker_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """For each column, list all marker genes that are NOT in that column."""
    full = pd.unique(mat.to_numpy().ravel())
    out = {}
    for col in mat.columns:
        excluded = set(mat[col].dropna())
        out[col] = [g for g in full if g not in excluded]
    max_len = max(len(v) for v in out.values())
    for col in out:
        out[col] = out[col] + [np.nan] * (max_len - len(out[col]))
    return pd.DataFrame(out)


def pos_neg_select(
    input: pd.DataFrame,
    ref_mat: pd.DataFrame,
    metadata,
    cluster_col: str = "cluster",
    cutoff_score: float | None = 0.5,
) -> pd.DataFrame:
    """Score clusters against a reference of positive/negative markers."""
    from .clustify import clustify

    padded = pd.concat(
        [input, pd.DataFrame([[0.01] * input.shape[1]], columns=input.columns, index=["clustifyr0"])]
    )
    res = clustify(
        padded,
        ref_mat,
        metadata=metadata,
        cluster_col=cluster_col,
        per_cell=True,
        verbose=False,
        query_genes=list(ref_mat.index),
    )
    res = res.fillna(0)

    res2 = average_clusters(res.T, metadata, cluster_col=cluster_col, if_log=False, output_log=False)
    res2 = res2.T

    if cutoff_score is not None:
        res2 = res2.copy()
        for col in res2.columns:
            maxr = res2[col].max()
            if maxr > 0.1:
                mask = (res2[col] > 0) & (res2[col] < cutoff_score * maxr)
                res2.loc[mask, col] = 0

    return res2


def gene_pct(matrix: pd.DataFrame, genelist: list, clusters, returning: str = "mean") -> pd.Series:
    """Fraction of cells expressing each gene in ``genelist``, aggregated per cluster."""
    genelist = [g for g in genelist if g in matrix.index]
    clusters = pd.Series(clusters).astype(object)
    clusters = clusters.where(~clusters.isna(), "orig.NA")

    agg = {"mean": np.mean, "min": np.min, "max": np.max}[returning]
    out = {}
    for cluster in clusters.unique():
        if not genelist:
            out[cluster] = np.nan
            continue
        cols = matrix.columns[(clusters == cluster).to_numpy()]
        sub = matrix.loc[genelist, cols]
        detected = (sub > 0).sum(axis=1) / sub.shape[1]
        out[cluster] = agg(detected.to_numpy())
    return pd.Series(out)


def gene_pct_markerm(
    matrix: pd.DataFrame,
    marker_m: pd.DataFrame,
    metadata,
    cluster_col: str | None = None,
    norm: str | float | None = None,
) -> pd.DataFrame:
    """``gene_pct`` for every marker set (column) in ``marker_m``."""
    cluster_info = _cluster_ids_from_metadata(matrix, metadata, cluster_col)

    cols = {}
    for col in marker_m.columns:
        genelist = [g for g in marker_m[col].dropna().tolist()]
        cols[col] = gene_pct(matrix, genelist, cluster_info)
    out = pd.DataFrame(cols)

    if norm is not None:
        if norm == "divide":
            out = out.div(out.max(axis=0), axis=1)
        elif norm == "diff":
            out = out.sub(out.max(axis=0), axis=1)
        else:
            out = out.sub(out.max(axis=0) * norm, axis=1)
            out = out.where(out <= 0, 1).where(out >= 0, 0)

    return out.fillna(0)


def ref_feature_select(
    mat: pd.DataFrame,
    n: int = 3000,
    mode: str = "var",
    rm_lowvar: bool = True,
) -> list:
    """Select the top ``n`` most variable (or most correlated) genes."""
    variances = mat.var(axis=1, ddof=1).sort_values(ascending=False)
    if rm_lowvar:
        half = len(variances) // 2
        top_half = variances.iloc[:half]
        mat = mat.loc[top_half.index]
        v2 = top_half
    else:
        v2 = variances

    if mode == "cor":
        cor_mat = mat.T.corr(method="spearman").to_numpy()
        np.fill_diagonal(cor_mat, 0)
        cor_mat = np.abs(cor_mat)
        score = pd.Series(np.nanmax(cor_mat, axis=1), index=mat.index)
        score = score.sort_values(ascending=False)
        return list(score.index[:n])

    return list(v2.index[:n])


def feature_select_pca(
    mat: pd.DataFrame | None = None,
    pcs: pd.DataFrame | None = None,
    n_pcs: int = 10,
    percentile: float = 0.99,
    if_log: bool = True,
) -> list:
    """Select genes with the largest loadings on up to ``n_pcs`` components.

    If fewer components are available, use all available components.
    """
    if pcs is None:
        data = mat if if_log else np.log(mat + 1)
        centered = data.T - data.T.mean(axis=0)
        _, _, vt = np.linalg.svd(centered.to_numpy(), full_matrices=False)
        pca = pd.DataFrame(vt.T, index=mat.index)
    else:
        pca = pcs

    genes = []
    for i in range(min(n_pcs, pca.shape[1])):
        loadings = pca.iloc[:, i].abs()
        cutoff = np.quantile(loadings.to_numpy(), percentile)
        genes.extend(loadings.index[loadings >= cutoff].tolist())
    return genes


def append_genes(gene_vector: list, ref_matrix: pd.DataFrame) -> pd.DataFrame:
    """Reindex ``ref_matrix`` to ``gene_vector``, zero-filling any missing genes."""
    missing = [g for g in gene_vector if g not in ref_matrix.index]
    if missing:
        zeros = pd.DataFrame(0, index=missing, columns=ref_matrix.columns)
        ref_matrix = pd.concat([ref_matrix, zeros])
    return ref_matrix.loc[gene_vector]


def check_raw_counts(counts_matrix: pd.DataFrame, max_log_value: float = 50) -> str:
    """Guess whether a matrix holds raw counts, normalized, or log-normalized data."""
    arr = counts_matrix.to_numpy()
    if np.all(arr == np.floor(arr)):
        return "raw counts"
    if arr.max() > max_log_value:
        return "normalized"
    if arr.min() < 0:
        raise ValueError("negative values detected, likely scaled data")
    return "log-normalized"


def make_comb_ref(ref_mat: pd.DataFrame, if_log: bool = True, sep: str = "_and_") -> pd.DataFrame:
    """Augment a reference with pairwise-averaged "combination" cell types."""
    from itertools import combinations

    mat = np.expm1(ref_mat) if if_log else ref_mat.copy()
    comb_cols = {}
    for a, b in combinations(mat.columns, 2):
        comb_cols[f"{a}{sep}{b}"] = mat[[a, b]].mean(axis=1)
    comb_mat = pd.DataFrame(comb_cols, index=mat.index)
    new_mat = pd.concat([mat, comb_mat], axis=1)
    if if_log:
        new_mat = np.log1p(new_mat)
    return new_mat


def downsample_matrix(
    mat: pd.DataFrame,
    n: int | None = None,
    keep_cluster_proportions: bool = True,
    metadata=None,
    cluster_col: str = "cluster",
    rng: np.random.Generator | None = None,
    *,
    frac: float | None = None,
    per_cluster: bool = False,
) -> pd.DataFrame:
    """Sample exactly n cells or floor(frac * total) cells without replacement.

    Proportional sampling uses largest remainders, with ties in first-seen
    cluster order. per_cluster=True instead requests n cells from each group.
    Indexed metadata is aligned by cell ID; plain vectors remain positional.
    """
    if (n is None) == (frac is None):
        raise ValueError("specify exactly one of n or frac")
    if n is not None and (not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 0):
        raise ValueError("n must be a nonnegative integer; use frac for a fraction")
    if frac is not None and (not isinstance(frac, (int, float, np.number)) or isinstance(frac, (bool, np.bool_))
                             or not np.isfinite(frac) or not 0 <= frac <= 1):
        raise ValueError("frac must be a finite number between 0 and 1")
    if per_cluster and (frac is not None or not keep_cluster_proportions):
        raise ValueError("per_cluster requires n and keep_cluster_proportions=True")
    if not mat.columns.is_unique or mat.columns.hasnans:
        raise ValueError("cell IDs must be unique and nonmissing")
    target = int(n) if n is not None else int(np.floor(len(mat.columns) * frac))
    if not per_cluster and target > len(mat.columns):
        raise ValueError("requested count exceeds available cells")
    rng = rng if rng is not None else np.random.default_rng()
    if not keep_cluster_proportions:
        return mat.loc[:, rng.choice(mat.columns.to_numpy(), size=target, replace=False)]
    if metadata is None:
        raise ValueError("metadata is required for sampling by cluster")
    cluster_ids = _cluster_ids_from_metadata(mat, metadata, cluster_col)
    if pd.isna(cluster_ids).any():
        raise ValueError("cluster assignments must be nonmissing")
    groups = {}
    for cell, group in zip(mat.columns, cluster_ids):
        groups.setdefault(group, []).append(cell)
    sizes = np.array([len(cells) for cells in groups.values()], dtype=int)
    if per_cluster:
        if (sizes < target).any():
            raise ValueError("requested count exceeds cells in a cluster")
        counts = np.full(len(sizes), target, dtype=int)
    elif len(sizes):
        quotas = sizes * (target / sizes.sum())
        counts = np.floor(quotas).astype(int)
        remainder = target - int(counts.sum())
        order = np.argsort(-(quotas - counts), kind="stable")
        counts[order[:remainder]] += 1
    else:
        counts = np.array([], dtype=int)
    chosen = []
    for cells, count in zip(groups.values(), counts):
        # Sampling positions preserves arbitrary cell-ID types.
        chosen.extend(cells[i] for i in rng.choice(len(cells), size=int(count), replace=False))
    return mat.loc[:, chosen]


def calc_distance(
    coord: pd.DataFrame,
    metadata,
    cluster_col: str = "cluster",
    collapse_to_cluster: bool = False,
) -> pd.DataFrame:
    """Per-cell (or per-cluster) minimum Euclidean distance to every cluster."""
    dist = squareform(pdist(coord.to_numpy()))
    distm = pd.DataFrame(dist, index=coord.index, columns=coord.index)

    res = average_clusters(
        distm, metadata, cluster_col=cluster_col, if_log=False, output_log=False, method="min"
    )
    if collapse_to_cluster:
        return average_clusters(
            res.T, metadata, cluster_col=cluster_col, if_log=False, output_log=False, method="min"
        )
    return res
