"""Preranked GSEA (gene set enrichment analysis).

Python port of clustifyr's R/run_fgsea.R (``run_gsea``) plus the GSEA-related
helpers in R/utils.R (``calculate_pathway_gsea``, ``gmt_to_list``).

fgsea itself is an R/C++ package with no direct Python dependency to lean on,
so the core weighted running-sum enrichment statistic is reimplemented here
directly (and validated to match ``fgsea:::calcGseaStat`` exactly). The
permutation testing scheme follows the same statistical design as the
original (pre-multilevel-splitting) GSEA/fgsea algorithm: for each gene set,
sample random gene sets of the same size from the ranking and use their
enrichment scores as a null distribution to estimate a p-value and NES.
Unlike fgsea's ``fgseaMultilevel``, permutations are not pooled across gene
sets of matching size, so results won't bit-for-bit match R's RNG, but the
same statistical target is estimated.
"""

from __future__ import annotations

import gzip
import re

import numpy as np
import pandas as pd


def calc_gsea_stat(ranks: np.ndarray, hit_positions: np.ndarray, gsea_param: float = 1) -> float:
    """Weighted running-sum enrichment statistic (matches ``fgsea:::calcGseaStat``).

    ``ranks`` are stat values already sorted descending. ``hit_positions`` are
    the (0-indexed) positions within ``ranks`` of the gene set's members.
    """
    sorted_positions = np.sort(hit_positions)
    s = sorted_positions.astype(float)
    r = ranks
    p = gsea_param
    m = len(s)
    n = len(r)
    if m == n:
        raise ValueError("GSEA statistic is not defined when all genes are selected")

    r_adj = np.abs(r[sorted_positions]) ** p
    nr = r_adj.sum()

    if nr == 0:
        r_cumsum = (np.arange(m) + 1) / m
    else:
        r_cumsum = np.cumsum(r_adj) / nr

    tops = r_cumsum - (s - np.arange(m)) / (n - m)
    bottoms = tops - (1 / m if nr == 0 else r_adj / nr)

    max_p = tops.max()
    min_p = bottoms.min()
    if max_p == -min_p:
        return 0.0
    return max_p if max_p > -min_p else min_p


def _prepare_ranks(stats: pd.Series) -> tuple[np.ndarray, dict]:
    values = stats.to_numpy(dtype=float)
    order = np.argsort(-values, kind="stable")
    sorted_ranks = values[order]
    gene_to_pos = {gene: pos for pos, gene in enumerate(stats.index.to_numpy()[order])}
    return sorted_ranks, gene_to_pos


def fgsea_simple(
    pathways: dict[str, list],
    stats: pd.Series,
    min_size: int = 1,
    max_size: int | None = None,
    n_perm: int = 1000,
    gsea_param: float = 1,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Preranked GSEA over a collection of gene sets against a ranking of genes.

    ``stats`` maps gene -> ranking metric (e.g. expression). Returns a
    DataFrame with columns ``pathway``, ``pval``, ``es``, ``nes``, ``size``.
    Nonfinite statistics are excluded from the gene universe; pathway genes
    are deduplicated. Sets with no overlap or covering the whole universe
    are skipped because their enrichment statistic is undefined.
    """
    rng = rng or np.random.default_rng()
    stats = stats.loc[np.isfinite(stats.to_numpy())]
    ranks, gene_to_pos = _prepare_ranks(stats)
    n = len(ranks)
    max_size = min(max_size, n - 1) if max_size is not None else n - 1

    rows = []
    perm_cache: dict[int, np.ndarray] = {}

    for name, genes in pathways.items():
        hit_positions = np.array([gene_to_pos[g] for g in dict.fromkeys(genes) if g in gene_to_pos], dtype=int)
        size = len(hit_positions)
        if size < min_size or size > max_size or size == 0:
            continue

        es = calc_gsea_stat(ranks, hit_positions, gsea_param=gsea_param)

        if size not in perm_cache:
            perm_es = np.empty(n_perm)
            for i in range(n_perm):
                sample_positions = rng.choice(n, size=size, replace=False)
                perm_es[i] = calc_gsea_stat(ranks, sample_positions, gsea_param=gsea_param)
            perm_cache[size] = perm_es
        perm_es = perm_cache[size]

        if es >= 0:
            pool = perm_es[perm_es >= 0]
            pval = (1 + np.sum(pool >= es)) / (1 + len(pool))
            nes = es / pool.mean() if len(pool) > 0 and pool.mean() != 0 else np.nan
        else:
            pool = perm_es[perm_es < 0]
            pval = (1 + np.sum(pool <= es)) / (1 + len(pool))
            nes = es / abs(pool.mean()) if len(pool) > 0 and pool.mean() != 0 else np.nan

        rows.append({"pathway": name, "pval": pval, "es": es, "nes": nes, "size": size})

    return pd.DataFrame(rows, columns=["pathway", "pval", "es", "nes", "size"])


def run_gsea(
    expr_mat: pd.DataFrame,
    query_genes: list | dict[str, list],
    cluster_ids=None,
    n_perm: int = 1000,
    per_cell: bool = False,
    scale: bool = False,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """GSEA of gene set(s) against per-cluster (or per-cell) expression.

    ``query_genes`` is a single gene list or a name -> gene-list dict. Returns
    a DataFrame indexed by cluster/cell, with columns ``pathway``, ``pval``,
    ``nes``.
    """
    from .clusters import average_clusters

    geneset_list = query_genes if isinstance(query_genes, dict) else {"query_genes": query_genes}

    if not per_cell and expr_mat.shape[1] != len(cluster_ids):
        raise ValueError("cluster_ids do not match number of cells (columns) in expr_mat")

    mat = expr_mat
    if scale:
        mean = mat.mean(axis=1)
        std = mat.std(axis=1, ddof=1)
        mat = mat.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)

    avg_mat = mat if per_cell else average_clusters(mat, cluster_ids)

    max_size = max(len(v) for v in geneset_list.values())
    rows = []
    for col in avg_mat.columns:
        stats = avg_mat[col]
        res = fgsea_simple(geneset_list, stats, min_size=1, max_size=max_size, n_perm=n_perm, rng=rng)
        found = set(res["pathway"])
        for _, row in res.iterrows():
            rows.append({"cell": col, "pathway": row["pathway"], "pval": row["pval"], "nes": row["nes"]})
        # Preserve the output shape for pathways with no valid enrichment
        # statistic (no overlap, or covering the entire finite gene universe).
        for name in geneset_list:
            if name not in found:
                rows.append({"cell": col, "pathway": name, "pval": np.nan, "nes": np.nan})

    return pd.DataFrame(rows).set_index("cell")


def calculate_pathway_gsea(
    mat: pd.DataFrame,
    pathway_list: dict[str, list],
    n_perm: int = 1000,
    scale: bool = True,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """NES matrix (cell types x pathways) from a cluster-averaged expression matrix."""
    out = {}
    for name, genes in pathway_list.items():
        res = run_gsea(mat, {name: genes}, n_perm=n_perm, scale=scale, per_cell=True, rng=rng)
        out[name] = res["nes"]
    return pd.DataFrame(out)


def plot_pathway_gsea(
    mat: pd.DataFrame,
    pathway_list: dict[str, list],
    n_perm: int = 1000,
    scale: bool = True,
    topn: int = 5,
    returning: str = "both",
    rng: np.random.Generator | None = None,
):
    """Heatmap of the top ``topn`` pathways (by NES) per cluster.

    ``returning`` is ``"both"`` (NES matrix + heatmap Axes), ``"plot"``, or
    ``"matrix"``.
    """
    from .classify import cor_to_call_topn
    from .plot import plot_cor_heatmap

    res = calculate_pathway_gsea(mat, pathway_list, n_perm=n_perm, scale=scale, rng=rng)
    top_calls = cor_to_call_topn(res, threshold=-np.inf, topn=topn)
    col_topn = list(pd.unique(top_calls["type"]))

    ax = plot_cor_heatmap(res.fillna(0)[col_topn], legend_title="NES")

    if returning == "both":
        return res, ax
    if returning == "plot":
        return ax
    return res


def gmt_to_list(
    path: str,
    cutoff: int = 0,
    sep: str | None = None,
) -> dict[str, list]:
    """Parse GMT rows as name, description, then tab-separated genes.

    Descriptions may be arbitrary text, including URLs or ``na``. An explicit
    ``sep`` retains the legacy custom-regex parsing behavior.
    """
    pathways = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.rstrip("\r\n")
            if not line.strip() or line.startswith("#"):
                continue
            if sep is None:
                fields = line.split("\t")
                if len(fields) < 3 or not fields[0]:
                    raise ValueError(f"invalid GMT row at line {line_number}: expected name, description, and genes")
                path_name, _, *genes = fields
            else:
                fields = re.split(sep, line, maxsplit=1)
                if len(fields) != 2:
                    raise ValueError(f"invalid GMT row at line {line_number}: separator not found")
                path_name, genes_str = fields
                genes = genes_str.split("\t")
            path_name = path_name.replace("REACTOME_", "")
            genes = [g for g in genes if g]
            if cutoff <= 0 or len(genes) >= cutoff:
                pathways[path_name] = genes
    return pathways
