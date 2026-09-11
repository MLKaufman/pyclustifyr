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


def _collapse_column(value, default):
    if value is False or value is None:
        return None
    if value is True:
        return default
    if isinstance(value, str) and value:
        return value
    raise ValueError("collapse_to_cluster must be False, None, True, or a grouping column name")


def _metadata_for_calls(metadata, ids, id_col, group_col):
    if metadata is None or group_col not in metadata.columns:
        raise ValueError("metadata must contain the collapse grouping column")
    # Indexed metadata is primary; a separate explicit ID column is also supported.
    if not pd.Index(ids).isin(metadata.index).all() and id_col != group_col and id_col in metadata:
        metadata = metadata.set_index(id_col, drop=False)
    if not metadata.index.is_unique or metadata.index.hasnans:
        raise ValueError("metadata cell IDs must be unique and nonmissing")
    if not pd.Index(ids).isin(metadata.index).all():
        raise ValueError("call cell IDs must match metadata")
    if metadata.loc[pd.Index(ids).unique(), group_col].isna().any():
        raise ValueError("collapse grouping labels must be nonmissing")
    return metadata


def cor_to_call(
    cor_mat: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    cluster_col: str | None = "cluster",
    collapse_to_cluster: str | bool = False,
    threshold: float | str = 0,
    rename_prefix: str | None = None,
    carry_r: bool = False,
) -> pd.DataFrame:
    """Take the best-scoring reference type per cluster/cell.

    Ties are marked with a ``-CLASH!`` suffix on the type, matching R's
    ``cor_to_call()``. Missing correlations never compete for a call; rows
    without valid scores are unassigned with a missing ``r``.
    """
    cluster_col = cluster_col or "cluster"
    group_col = _collapse_column(collapse_to_cluster, cluster_col)
    if group_col is not None:
        metadata = _metadata_for_calls(metadata, cor_mat.index, cluster_col, group_col)
    missing_rows = cor_mat.index[cor_mat.isna().all(axis=1)]
    correlation_matrix = cor_mat.fillna(-np.inf)

    if threshold == "auto":
        values = cor_mat.to_numpy()
        threshold = round(0.75 * float(np.nanmax(values)), 2) if (~pd.isna(values)).any() else 0

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

    missing = best[cluster_col].isin(missing_rows)
    best.loc[missing, "type"] = unassigned_label
    best.loc[missing, "r"] = np.nan
    result = best

    if group_col is not None:
        result = collapse_to_cluster_fn(result, metadata, group_col, threshold=threshold)
        if carry_r:
            rejected = result["n"] == 0
            result.loc[rejected, "type"] = unassigned_label

    if rename_prefix is not None:
        if group_col is not None:
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
    metadata = _metadata_for_calls(metadata, res.iloc[:, 0], res.columns[0], cluster_col)
    df = res.copy()
    # Use temporary column names independent of the requested grouping column.
    df["_group"] = df.iloc[:, 0].map(metadata[cluster_col])
    groups = pd.Index(df["_group"].drop_duplicates(), name=cluster_col)
    eligible = df[df["r"].notna() & (df["r"] >= threshold)]
    grouped = eligible.groupby(["type", "_group"], sort=False, observed=True)["r"].agg(sum="sum", n="count").reset_index()
    grouped = grouped.sort_values(["n", "sum"], ascending=[False, False], kind="stable")
    top = grouped.groupby("_group", sort=False).head(1).set_index("_group")
    top = top.reindex(groups)
    top["type"] = top["type"].fillna("unassigned")
    top["n"] = top["n"].fillna(0).astype(int)
    return top.rename_axis(cluster_col).reset_index()[[cluster_col, "type", "sum", "n"]]


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
    group_col = _collapse_column(collapse_to_cluster, col)
    if group_col is not None:
        metadata = _metadata_for_calls(metadata, cor_mat.index, col, group_col)
    long = _melt_cor_mat(cor_mat, col)
    unassigned_label = f"r<{threshold}, unassigned"
    long.loc[long["r"] < threshold, "type"] = unassigned_label

    rank = long.groupby(col)["r"].rank(method="min", ascending=False)
    top = long[rank <= topn].reset_index(drop=True)

    if group_col is not None:
        merged = top.copy()
        for name in metadata.columns:
            if name not in merged.columns:
                merged[name] = merged[col].map(metadata[name])
        merged["type2"] = merged[col].map(metadata[group_col])
        eligible = merged[merged["r"].notna() & (merged["r"] >= threshold)]
        grouped = eligible.groupby(["type", "type2"], sort=False, observed=True)["r"].agg(sum="sum", n="count").reset_index()
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
    """Left-join a call table onto metadata.

    Per-cell calls use the metadata index as the cell identifier; cluster
    calls use ``cluster_col``. Metadata row order is preserved. Existing
    prediction columns with the same output names are replaced.
    """
    df_temp = res.copy()
    if rename_prefix is not None:
        df_temp = df_temp.rename(columns={
            "type": f"{rename_prefix}_type", "r": f"{rename_prefix}_r",
            "rank": f"{rename_prefix}_rank",
        })

    # Replace previous predictions, without changing the caller's metadata.
    metadata = metadata.drop(columns=list(df_temp.columns[1:]), errors="ignore")
    if not per_cell:
        if cluster_col not in metadata.columns:
            raise ValueError("cluster_col is not a column of metadata")
        if cluster_col not in df_temp.columns:
            raise ValueError("cluster_col is not a column of called cell type dataframe")
        original_clusters = metadata[cluster_col]
        metadata = metadata.copy()
        metadata[cluster_col] = original_clusters.astype(object).where(original_clusters.notna(), "orig.NA")
        if not set(df_temp[cluster_col].unique()).issubset(set(metadata[cluster_col].unique())):
            raise ValueError("cluster_col from clustify step and joining to metadata step are not the same")
        merged = metadata.merge(df_temp, on=cluster_col, how="left", suffixes=("", ".clustify"))
        merged.index = metadata.index
        merged[cluster_col] = original_clusters
        return merged
    else:
        calls = df_temp.set_index(df_temp.columns[0])
        return metadata.join(calls, how="left", rsuffix=".clustify", validate="many_to_one")


def call_consensus(list_of_res: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple ``cor_to_call_rank`` outputs into a consensus call."""
    res = pd.concat(list_of_res, ignore_index=True)
    cols = list(res.columns[:2])
    rank_col = "rank"

    grouped = res.groupby(cols, sort=False)[rank_col].mean().reset_index()
    best_rank = grouped.groupby(cols[0], sort=False)[rank_col].transform("min")
    grouped = grouped[grouped[rank_col] == best_rank]

    combined = (
        grouped.groupby([cols[0], rank_col], sort=False)[cols[1]]
        .agg(lambda s: "__".join(s))
        .reset_index()
    )
    return combined[[cols[0], cols[1], rank_col]]
