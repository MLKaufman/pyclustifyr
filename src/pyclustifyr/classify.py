"""Turn similarity matrices into cell-type calls.

Python port of clustifyr's R/common_dplyr.R and R/utils.R (cor_to_call*,
call_to_metadata, collapse_to_cluster, call_consensus).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _melt_cor_mat(cor_mat: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    df = cor_mat.copy()
    df.index.name = cluster_col
    long = df.reset_index().melt(id_vars=cluster_col, var_name="type", value_name="r")
    return long


def cor_to_call(
    cor_mat: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    cluster_col: str = "cluster",
    collapse_to_cluster: str | bool = False,
    threshold: float | str = 0,
    rename_prefix: str | None = None,
    carry_r: bool = False,
) -> pd.DataFrame:
    """Take the best-scoring reference type per cluster/cell.

    Ties are marked with a ``-CLASH!`` suffix on the type, matching R's
    ``cor_to_call()``.
    """
    correlation_matrix = cor_mat.fillna(0)

    if threshold == "auto":
        threshold = round(0.75 * float(np.nanmax(correlation_matrix.to_numpy())), 2)

    long = _melt_cor_mat(correlation_matrix, cluster_col)

    unassigned_label = f"r<{threshold}, unassigned" if carry_r else "unassigned"
    long.loc[long["r"] < threshold, "type"] = unassigned_label

    group_max = long.groupby(cluster_col)["r"].transform("max")
    best = long[long["r"] == group_max].reset_index(drop=True)

    if len(best) != len(correlation_matrix):
        counts = best.groupby(cluster_col)[cluster_col].transform("size")
        clash_mask = counts > 1
        best.loc[clash_mask, "type"] = best.loc[clash_mask, "type"] + "-CLASH!"
        best = best.drop_duplicates(subset=[c for c in best.columns if c != "type"])

    result = best

    if collapse_to_cluster is not False:
        if metadata is None:
            raise ValueError("metadata is required when collapse_to_cluster is set")
        result = collapse_to_cluster_fn(result, metadata, cluster_col, threshold=threshold)

    if rename_prefix is not None:
        if collapse_to_cluster is not False:
            result = result.rename(
                columns={"type": f"{rename_prefix}_type", "sum": f"{rename_prefix}_sum", "n": f"{rename_prefix}_n"}
            )
        else:
            result = result.rename(columns={"type": f"{rename_prefix}_type", "r": f"{rename_prefix}_r"})

    return result.reset_index(drop=True)


def collapse_to_cluster_fn(
    res: pd.DataFrame,
    metadata: pd.DataFrame,
    cluster_col: str,
    threshold: float = 0,
) -> pd.DataFrame:
    """From per-cell calls, take the highest-frequency call within each cluster."""
    df = res.copy()
    df = df.rename(columns={df.columns[0]: "rn"})
    df["cluster"] = list(metadata[cluster_col])

    grouped = df.groupby(["type", "cluster"], sort=False)["r"].agg(sum="sum", n="count").reset_index()
    grouped = grouped[grouped["type"] != f"r<{threshold}, unassigned"]
    grouped = grouped.sort_values(["n", "sum"], ascending=[False, False])
    top = grouped.groupby("cluster", sort=False).head(1)
    top = top.rename(columns={"cluster": cluster_col})
    return top[[cluster_col, "type", "sum", "n"]].reset_index(drop=True)


# Public alias matching clustifyr's exported name.
collapse_to_cluster = collapse_to_cluster_fn


def cor_to_call_rank(
    cor_mat: pd.DataFrame,
    cluster_col: str = "cluster",
    threshold: float | str = 0,
    rename_prefix: str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Rank every reference type per cluster/cell by descending similarity."""
    if threshold == "auto":
        threshold = round(0.75 * float(np.nanmax(cor_mat.to_numpy())), 2)

    long = _melt_cor_mat(cor_mat, cluster_col)
    long["rank"] = long.groupby(cluster_col)["r"].rank(method="dense", ascending=False)
    long.loc[long["r"] < threshold, "rank"] = 100

    if top_n is not None:
        long = long[long["rank"] <= top_n]

    if rename_prefix is not None:
        long = long.rename(columns={"type": f"{rename_prefix}_type", "r": f"{rename_prefix}_r"})

    return long.reset_index(drop=True)


def cor_to_call_topn(
    cor_mat: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    col: str = "cluster",
    collapse_to_cluster: str | bool = False,
    threshold: float = 0,
    topn: int = 2,
) -> pd.DataFrame:
    """Take the top ``topn`` reference-type calls per cluster/cell."""
    long = _melt_cor_mat(cor_mat, col)
    unassigned_label = f"r<{threshold}, unassigned"
    long.loc[long["r"] < threshold, "type"] = unassigned_label

    rank = long.groupby(col)["r"].rank(method="min", ascending=False)
    top = long[rank <= topn].reset_index(drop=True)

    if collapse_to_cluster is not False:
        if metadata is None:
            raise ValueError("metadata is required when collapse_to_cluster is set")
        merged = top.merge(metadata, on=col, how="left")
        merged["type2"] = merged[collapse_to_cluster]
        grouped = merged.groupby(["type", "type2"], sort=False)["r"].agg(sum="sum", n="count").reset_index()
        grouped = grouped[grouped["type"] != unassigned_label]
        grouped = grouped.sort_values(["n", "sum"], ascending=[False, False])
        top_per_type2 = grouped.groupby("type2", sort=False).head(topn)
        joined = top_per_type2.merge(
            merged.drop(columns=["type", "r"]),
            on="type2",
            how="right",
        )
        joined["type"] = joined["type"].fillna(unassigned_label)
        joined = joined.drop_duplicates(subset=[col, "type", "type2"])
        return joined.sort_values(["n", "sum"], ascending=[False, False]).reset_index(drop=True)

    return top.sort_values([col, "r"], ascending=[True, False]).reset_index(drop=True)


def call_to_metadata(
    res: pd.DataFrame,
    metadata: pd.DataFrame,
    cluster_col: str,
    per_cell: bool = False,
    rename_prefix: str | None = None,
) -> pd.DataFrame:
    """Left-join a call table (e.g. output of ``cor_to_call``) onto metadata."""
    df_temp = res.copy()
    if rename_prefix is not None:
        df_temp = df_temp.rename(columns={"type": f"{rename_prefix}_type", "r": f"{rename_prefix}_r"})

    if not per_cell:
        if cluster_col not in metadata.columns:
            raise ValueError("cluster_col is not a column of metadata")
        if cluster_col not in df_temp.columns:
            raise ValueError("cluster_col is not a column of called cell type dataframe")
        if not set(df_temp[cluster_col].unique()).issubset(set(metadata[cluster_col].unique())):
            raise ValueError("cluster_col from clustify step and joining to metadata step are not the same")
        merged = metadata.merge(df_temp, on=cluster_col, how="left", suffixes=("", ".clustify"))
        merged.index = metadata.index
        return merged
    else:
        df_temp = df_temp.rename(columns={df_temp.columns[0]: cluster_col})
        merged = metadata.merge(df_temp, on=cluster_col, how="left", suffixes=("", ".clustify"))
        merged.index = metadata.index
        return merged


def call_consensus(list_of_res: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple ``cor_to_call_rank`` outputs into a consensus call."""
    res = pd.concat(list_of_res, ignore_index=True)
    cols = list(res.columns[:2])
    rank_col = res.columns[2]

    grouped = res.groupby(cols, sort=False)[rank_col].mean().reset_index()
    best_rank = grouped.groupby(cols[0], sort=False)[rank_col].transform("min")
    grouped = grouped[grouped[rank_col] == best_rank]

    combined = (
        grouped.groupby([cols[0], rank_col], sort=False)[cols[1]]
        .agg(lambda s: "__".join(s))
        .reset_index()
    )
    return combined[[cols[0], cols[1], rank_col]]
