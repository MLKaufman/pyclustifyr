"""Similarity/correlation scoring between query and reference expression matrices.

Python port of clustifyr's R/compute_similarity.R.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

clustifyr_methods = ("pearson", "spearman", "cosine", "kl_divergence", "kendall")


def _pearson_cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation between columns of x and columns of y."""
    xc = x - np.nanmean(x, axis=0, keepdims=True)
    yc = y - np.nanmean(y, axis=0, keepdims=True)
    numer = xc.T @ yc
    denom = np.sqrt((xc**2).sum(axis=0))[:, None] * np.sqrt((yc**2).sum(axis=0))[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return numer / denom


def _spearman_cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xr = rankdata(x, axis=0)
    yr = rankdata(y, axis=0)
    return _pearson_cross(xr, yr)


def _kendall_cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_x, n_y = x.shape[1], y.shape[1]
    out = np.empty((n_x, n_y))
    for i in range(n_x):
        for j in range(n_y):
            out[i, j] = kendalltau(x[:, i], y[:, j]).statistic
    return out


def _cosine_cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Column-wise cosine similarity between columns of x and columns of y."""
    numer = x.T @ y
    x_norm = np.sqrt((x**2).sum(axis=0))[:, None]
    y_norm = np.sqrt((y**2).sum(axis=0))[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return numer / (x_norm * y_norm)


def cosine(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.sum(vec1 * vec2) / np.sqrt(np.sum(vec1**2) * np.sum(vec2**2)))


def kl_divergence(
    vec1: np.ndarray,
    vec2: np.ndarray,
    if_log: bool = False,
    total_reads: int = 1000,
    max_kl: float = 1,
) -> float:
    """Similarity score derived from Kullback-Leibler divergence.

    Reads are rescaled to a pseudo-library size of ``total_reads``, rounded to
    pseudo-counts, and KL-divergence (with shrinkage) is linearly rescaled from
    [0, max_kl] to a similarity score in [-1, 1]. The result is not clipped;
    divergence above ``max_kl`` produces a score below -1.
    """
    vec1 = np.asarray(vec1, dtype=float)
    vec2 = np.asarray(vec2, dtype=float)
    if if_log:
        vec1 = np.expm1(vec1)
        vec2 = np.expm1(vec2)
    count1 = np.round(vec1 * total_reads / vec1.sum())
    count2 = np.round(vec2 * total_reads / vec2.sum())
    est_kl = _kl_shrink(count1, count2)
    return (max_kl - est_kl) / max_kl * 2 - 1


def _kl_shrink(count1: np.ndarray, count2: np.ndarray) -> float:
    """Shrinkage-based KL divergence estimate, matching entropy::KL.shrink.

    Uses the James-Stein shrinkage estimator of Hausser & Strimmer for the
    underlying frequency distributions before computing KL divergence in nats.
    """
    p1 = _freqs_shrink(count1)
    p2 = _freqs_shrink(count2)
    mask = p1 > 0
    return float(np.sum(p1[mask] * np.log(p1[mask] / p2[mask])))


def _freqs_shrink(counts: np.ndarray) -> np.ndarray:
    """James-Stein shrinkage frequency estimator (entropy::freqs.shrink)."""
    n = counts.sum()
    k = counts.size
    target = 1.0 / k

    if n == 0 or n == 1:
        return np.full(k, target)

    u = counts / n
    numer = 1 - np.sum(u**2)
    denom = (n - 1) * np.sum((target - u) ** 2)
    lam = 1.0 if denom == 0 else numer / denom
    lam = min(1.0, max(0.0, lam))
    return lam * target + (1 - lam) * u


def vector_similarity(vec1: np.ndarray, vec2: np.ndarray, compute_method: str, **kwargs) -> float:
    vec1 = np.asarray(vec1, dtype=float)
    vec2 = np.asarray(vec2, dtype=float)
    if vec1.shape != vec2.shape:
        raise ValueError(
            "compute_similarity: two input vectors are not numeric or of different sizes."
        )
    if compute_method not in ("cosine", "kl_divergence"):
        raise ValueError(f"{compute_method} not implemented")
    if compute_method == "kl_divergence":
        return kl_divergence(vec1, vec2, **kwargs)
    return cosine(vec1, vec2)


def calc_similarity(
    query_mat: pd.DataFrame,
    ref_mat: pd.DataFrame,
    compute_method: str,
    rm0: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Compute a similarity/correlation matrix between columns of two matrices.

    ``rm0`` treats query zeros as missing and uses pairwise complete
    observations. It supports Pearson, Spearman, and Kendall correlations.
    """
    sc_clust = list(query_mat.columns)
    ref_clust = list(ref_mat.columns)

    if rm0:
        if compute_method not in ("pearson", "spearman", "kendall"):
            raise ValueError("rm0 is supported only for pearson, spearman, and kendall")
        q = query_mat.to_numpy(dtype=float).copy()
        q[q == 0] = np.nan
        r = ref_mat.to_numpy(dtype=float)
        if compute_method == "spearman":
            score = _pairwise_complete_spearman(q, r)
        elif compute_method == "kendall":
            score = _pairwise_complete_kendall(q, r)
        else:
            score = _pairwise_complete_pearson(q, r)
        return pd.DataFrame(score, index=sc_clust, columns=ref_clust)

    if compute_method in ("pearson", "spearman", "kendall"):
        x = query_mat.to_numpy(dtype=float)
        y = ref_mat.to_numpy(dtype=float)
        if compute_method == "pearson":
            score = _pearson_cross(x, y)
        elif compute_method == "spearman":
            score = _spearman_cross(x, y)
        else:
            score = _kendall_cross(x, y)
        return pd.DataFrame(score, index=sc_clust, columns=ref_clust)

    if compute_method == "cosine":
        x = query_mat.to_numpy(dtype=float)
        y = ref_mat.to_numpy(dtype=float)
        score = _cosine_cross(x, y)
        return pd.DataFrame(score, index=sc_clust, columns=ref_clust)

    features = query_mat.index.intersection(ref_mat.index)
    q = query_mat.loc[features]
    r = ref_mat.loc[features]
    score = np.full((len(sc_clust), len(ref_clust)), np.nan)
    for i, sc in enumerate(sc_clust):
        for j, rc in enumerate(ref_clust):
            score[i, j] = vector_similarity(
                q[sc].to_numpy(dtype=float), r[rc].to_numpy(dtype=float), compute_method, **kwargs
            )
    return pd.DataFrame(score, index=sc_clust, columns=ref_clust)


def _pairwise_complete_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_x, n_y = x.shape[1], y.shape[1]
    out = np.full((n_x, n_y), np.nan)
    for i in range(n_x):
        for j in range(n_y):
            mask = ~np.isnan(x[:, i]) & ~np.isnan(y[:, j])
            xv, yv = x[mask, i], y[mask, j]
            if xv.size < 2:
                continue
            out[i, j] = np.corrcoef(xv, yv)[0, 1]
    return out


def _pairwise_complete_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_x, n_y = x.shape[1], y.shape[1]
    out = np.full((n_x, n_y), np.nan)
    for i in range(n_x):
        for j in range(n_y):
            mask = ~np.isnan(x[:, i]) & ~np.isnan(y[:, j])
            xv, yv = x[mask, i], y[mask, j]
            if xv.size < 2:
                continue
            out[i, j] = np.corrcoef(rankdata(xv), rankdata(yv))[0, 1]
    return out


def _pairwise_complete_kendall(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full((x.shape[1], y.shape[1]), np.nan)
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            mask = ~np.isnan(x[:, i]) & ~np.isnan(y[:, j])
            if mask.sum() >= 2:
                out[i, j] = kendalltau(x[mask, i], y[mask, j]).statistic
    return out


def get_similarity(
    expr_mat: pd.DataFrame,
    ref_mat: pd.DataFrame,
    cluster_ids,
    compute_method: str,
    pseudobulk_method: str = "mean",
    per_cell: bool = False,
    rm0: bool = False,
    if_log: bool = True,
    low_threshold: int = 0,
    **kwargs,
) -> pd.DataFrame:
    from .clusters import average_clusters

    if expr_mat.shape[0] == 0:
        raise ValueError("after subsetting to shared genes, query expression matrix has 0 rows")
    if expr_mat.shape[1] == 0:
        raise ValueError("query expression matrix has 0 cols")
    if ref_mat.shape[0] == 0:
        raise ValueError("after subsetting to shared genes, reference expression matrix has 0 rows")
    if ref_mat.shape[1] == 0:
        raise ValueError("reference expression matrix has 0 cols")

    ref_clust = list(ref_mat.columns)
    cluster_ids = list(cluster_ids)
    if expr_mat.shape[1] != len(cluster_ids):
        raise ValueError("number of cells in expression matrix not equal to metadata/cluster_col")

    cluster_ids = ["unknown" if pd.isna(c) else c for c in cluster_ids]

    if not per_cell:
        clust_avg = average_clusters(
            expr_mat,
            cluster_ids,
            if_log=if_log,
            low_threshold=low_threshold,
            method=pseudobulk_method,
        )
        sc_clust = list(clust_avg.columns)
    else:
        sc_clust = cluster_ids
        clust_avg = expr_mat

    assigned_score = calc_similarity(clust_avg, ref_mat, compute_method, rm0=rm0, **kwargs)

    if low_threshold == 0:
        assigned_score.index = sc_clust
        assigned_score.columns = ref_clust

    return assigned_score


def permute_similarity(
    expr_mat: pd.DataFrame,
    ref_mat: pd.DataFrame,
    cluster_ids,
    n_perm: int,
    per_cell: bool = False,
    compute_method: str = "spearman",
    pseudobulk_method: str = "mean",
    rm0: bool = False,
    rng: np.random.Generator | None = None,
    if_log: bool = True,
    low_threshold: int = 0,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Estimate upper-tail p-values as (1 + count(null >= observed)) / (n_perm + 1).

    Undefined observed or null scores produce missing p-values.
    """
    from .clusters import average_clusters

    if not isinstance(n_perm, (int, np.integer)) or n_perm < 1:
        raise ValueError("n_perm must be a positive integer")
    rng = rng or np.random.default_rng()
    ref_clust = list(ref_mat.columns)
    cluster_ids = list(cluster_ids)

    if not per_cell:
        clust_avg = average_clusters(
            expr_mat, cluster_ids, method=pseudobulk_method,
            if_log=if_log, low_threshold=low_threshold,
        )
        sc_clust = list(clust_avg.columns)
    else:
        sc_clust = list(expr_mat.columns)
        clust_avg = expr_mat

    assigned_score = calc_similarity(clust_avg, ref_mat, compute_method, rm0=rm0, **kwargs)

    sig_counts = np.zeros((len(sc_clust), len(ref_clust)), dtype=int)
    cluster_ids_arr = np.array(cluster_ids, dtype=object)
    observed = assigned_score.to_numpy()
    valid_scores = ~np.isnan(observed)

    for _ in range(n_perm):
        resampled = rng.permutation(cluster_ids_arr)
        if not per_cell:
            permuted_avg = average_clusters(
                expr_mat, list(resampled), method=pseudobulk_method,
                if_log=if_log, low_threshold=low_threshold,
            ).reindex(columns=sc_clust)
        else:
            permuted_avg = expr_mat.loc[:, resampled]

        new_score = calc_similarity(permuted_avg, ref_mat, compute_method, rm0=rm0, **kwargs)
        null_scores = new_score.to_numpy()
        valid_scores &= ~np.isnan(null_scores)
        sig_counts += (null_scores >= observed).astype(int)

    assigned_score.index = sc_clust
    assigned_score.columns = ref_clust
    pvalues = (sig_counts + 1) / (n_perm + 1)
    pvalues[~valid_scores] = np.nan
    p_val = pd.DataFrame(pvalues, index=sc_clust, columns=ref_clust)

    return {"score": assigned_score, "p_val": p_val}
