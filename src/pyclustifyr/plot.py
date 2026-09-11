"""Plotting helpers for correlation/expression/call results.

Python port of clustifyr's R/plot.R, using matplotlib in place of
ggplot2/cowplot/ComplexHeatmap. Each R function that returned a single
ggplot object returns a matplotlib ``Axes`` here; functions that returned a
list of ggplot objects (one per column/gene) return a list of ``Axes``.

Note: R's ``do_repel=True`` uses ggrepel to nudge overlapping cluster labels
apart. There's no lightweight matplotlib equivalent bundled here, so labels
are placed directly at each cluster's centroid regardless of ``do_repel``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .classify import call_to_metadata, collapse_to_cluster as collapse_to_cluster_fn, cor_to_call

# Continuous palette for correlation/expression values (matches clustifyr's pretty_palette2).
pretty_palette2 = "Reds"
# Continuous diverging palette (matches clustifyr's pretty_palette).
pretty_palette = "RdGy_r"
# Grayscale palette for heatmaps (matches clustifyr's not_pretty_palette).
not_pretty_palette = "Greys"
# Qualitative palette for discrete features (matches clustifyr's default discrete brewer palette).
discrete_palette = "tab20"


def _discrete_colors(n: int, cmap_name: str = discrete_palette) -> list:
    cmap = plt.get_cmap(cmap_name)
    if hasattr(cmap, "colors") and n <= len(cmap.colors):
        return [cmap.colors[i] for i in range(n)]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_dims(
    data: pd.DataFrame,
    x: str = "UMAP_1",
    y: str = "UMAP_2",
    feature: str | None = None,
    legend_name: str | None = None,
    cmap: str = pretty_palette2,
    d_cols: dict | list | None = None,
    pt_size: float = 5,
    alpha_col: str | None = None,
    group_col: str | None = None,
    scale_limits: tuple[float, float] | None = None,
    do_label: bool = False,
    do_legend: bool = True,
    do_repel: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Scatter plot of cells in a 2D embedding (e.g. UMAP), colored by ``feature``."""
    if ax is None:
        _, ax = plt.subplots()
    legend_name = legend_name if legend_name is not None else (feature or "")

    if feature is None:
        ax.scatter(data[x], data[y], s=pt_size)
        return ax

    data = data.sort_values(feature)
    alpha = None
    if alpha_col is not None:
        vals = data[alpha_col].to_numpy(dtype=float)
        span = vals.max() - vals.min()
        alpha = (vals - vals.min()) / span if span > 0 else np.ones_like(vals)

    feature_vals = data[feature]
    is_numeric = pd.api.types.is_numeric_dtype(feature_vals) and not pd.api.types.is_bool_dtype(feature_vals)

    if is_numeric:
        if scale_limits is None:
            vmin = min(0, feature_vals.min())
            vmax = feature_vals.max()
        else:
            vmin, vmax = scale_limits
        sc = ax.scatter(data[x], data[y], c=feature_vals, cmap=cmap, vmin=vmin, vmax=vmax, s=pt_size, alpha=alpha)
        if do_legend:
            plt.colorbar(sc, ax=ax, label=legend_name)
    else:
        categories = list(pd.unique(feature_vals))
        if isinstance(d_cols, dict):
            colors = d_cols
        else:
            palette = d_cols if d_cols is not None else _discrete_colors(len(categories))
            colors = dict(zip(categories, palette))
        for cat in categories:
            mask = (feature_vals == cat).to_numpy()
            a = alpha[mask] if alpha is not None else None
            ax.scatter(
                data[x].to_numpy()[mask],
                data[y].to_numpy()[mask],
                color=colors[cat],
                s=pt_size,
                alpha=a,
                label=str(cat),
            )
        if do_legend:
            ax.legend(title=legend_name, loc="best", fontsize="small", markerscale=2)

    if do_label:
        group_cols = [feature] if group_col is None else [feature, group_col]
        centers = data.groupby(group_cols, observed=True)[[x, y]].median()
        for idx, row in centers.iterrows():
            label = idx[0] if isinstance(idx, tuple) else idx
            ax.annotate(str(label), (row[x], row[y]), ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def plot_cor(
    cor_mat: pd.DataFrame,
    metadata: pd.DataFrame,
    data_to_plot: list | None = None,
    cluster_col: str | None = None,
    x: str = "UMAP_1",
    y: str = "UMAP_2",
    scale_legends: bool | tuple[float, float] = False,
    **kwargs,
) -> list[Axes]:
    """Plot similarity scores for each reference type, one embedding plot per column."""
    data_to_plot = data_to_plot if data_to_plot is not None else list(cor_mat.columns)
    if not any(c in cor_mat.columns for c in data_to_plot):
        raise ValueError("cluster ids not shared between metadata and correlation matrix")

    meta = metadata.copy()
    join_col = cluster_col
    if join_col is None:
        join_col = "rownames"
        meta[join_col] = meta.index

    cor_df = cor_mat.copy()
    cor_df[join_col] = cor_df.index
    long = cor_df.melt(id_vars=join_col, var_name="ref_cluster", value_name="expr")

    if long[join_col].iloc[0] in set(meta[join_col]):
        plt_data_full = long.merge(meta, on=join_col, how="left")
    else:
        plt_data_full = long.merge(meta, left_on=join_col, right_index=True, how="left")

    if isinstance(scale_legends, bool):
        if scale_legends:
            vmin = min(0, plt_data_full["expr"].min())
            vmax = plt_data_full["expr"].max()
            scale_limits = (vmin, vmax)
        else:
            scale_limits = None
    else:
        scale_limits = scale_legends

    axes = []
    for col in data_to_plot:
        subset = plt_data_full[plt_data_full["ref_cluster"] == col]
        ax = plot_dims(subset, x=x, y=y, feature="expr", legend_name=col, scale_limits=scale_limits, **kwargs)
        axes.append(ax)
    return axes


def plot_gene(
    expr_mat: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: list,
    cell_col: str | None = None,
    **kwargs,
) -> list[Axes]:
    """Plot expression of one or more genes on a 2D embedding."""
    genes_to_plot = [g for g in genes if g in expr_mat.index]
    missing = [g for g in genes if g not in genes_to_plot]
    if missing:
        print(f"the following genes were not present in the input matrix {','.join(missing)}")
    if not genes_to_plot:
        raise ValueError("no genes present to plot")

    expr_df = expr_mat.loc[genes_to_plot].T
    meta = metadata if cell_col is not None else metadata.copy()
    if cell_col is None:
        meta = meta.copy()
        meta["cell"] = meta.index
        cell_col = "cell"
    elif cell_col not in meta.columns:
        raise ValueError("please supply a cell_col that is present in metadata")

    plt_data = expr_df.merge(meta, left_index=True, right_on=cell_col, how="left")
    return [plot_dims(plt_data, feature=gene, legend_name=gene, **kwargs) for gene in genes_to_plot]


def plot_best_call(
    cor_mat: pd.DataFrame,
    metadata: pd.DataFrame,
    cluster_col: str = "cluster",
    collapse_to_cluster: str | bool = False,
    threshold: float = 0,
    x: str = "UMAP_1",
    y: str = "UMAP_2",
    plot_r: bool = False,
    per_cell: bool = False,
    **kwargs,
) -> Axes | list[Axes]:
    """Plot the best-scoring cell-type call for each cluster/cell on a 2D embedding."""
    if "type" in metadata.columns or "type2" in metadata.columns:
        raise ValueError('metadata column name clash of "type"/"type2"')

    df_temp = cor_to_call(cor_mat, metadata=metadata, cluster_col=cluster_col, threshold=threshold)
    df_full = call_to_metadata(df_temp, metadata=metadata, cluster_col=cluster_col, per_cell=per_cell)

    if collapse_to_cluster is not False:
        target_col = cluster_col if collapse_to_cluster is True else collapse_to_cluster
        cell_calls = df_full[["type", "r"]].copy()
        cell_calls.insert(0, "cell_id", df_full.index)
        collapsed = collapse_to_cluster_fn(cell_calls, metadata, target_col, threshold=threshold)
        # Keep the per-cell score for plot_r, but color every cell by its
        # cluster's majority call while retaining embedding coordinates.
        df_full["type"] = df_full[target_col].map(collapsed.set_index(target_col)["type"])
        df_full["type"] = df_full["type"].fillna("unassigned")

    g = plot_dims(df_full, feature="type", x=x, y=y, **kwargs)
    if plot_r:
        return [g, plot_dims(df_full, feature="r", x=x, y=y, **kwargs)]
    return g


def plot_cor_heatmap(
    cor_mat: pd.DataFrame,
    cmap: str = not_pretty_palette,
    legend_title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Heatmap of a similarity matrix (clusters x reference types)."""
    if ax is None:
        fig_width = max(4, 0.4 * cor_mat.shape[1])
        fig_height = max(3, 0.3 * cor_mat.shape[0])
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.figure

    im = ax.imshow(cor_mat.to_numpy(), cmap=cmap, aspect="auto")
    ax.set_xticks(range(cor_mat.shape[1]))
    ax.set_xticklabels(cor_mat.columns, rotation=90)
    ax.set_yticks(range(cor_mat.shape[0]))
    ax.set_yticklabels(cor_mat.index)
    fig.colorbar(im, ax=ax, label=legend_title)
    return ax
