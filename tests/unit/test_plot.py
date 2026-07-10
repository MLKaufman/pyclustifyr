import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes

from pyclustifyr.clustify import clustify
from pyclustifyr.plot import plot_best_call, plot_cor, plot_cor_heatmap, plot_dims, plot_gene


@pytest.fixture
def cor_res(pbmc_matrix_small, pbmc_meta, cbmc_ref):
    return clustify(pbmc_matrix_small, cbmc_ref, metadata=pbmc_meta, cluster_col="classified", verbose=False)


def test_plot_dims_discrete(pbmc_meta):
    ax = plot_dims(pbmc_meta, feature="classified", do_label=True)
    assert isinstance(ax, Axes)
    assert len(ax.collections) == pbmc_meta["classified"].nunique()
    plt.close(ax.figure)


def test_plot_dims_continuous(pbmc_meta):
    ax = plot_dims(pbmc_meta, feature="percent.mt")
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_plot_dims_no_feature(pbmc_meta):
    ax = plot_dims(pbmc_meta)
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_plot_cor(cor_res, pbmc_meta):
    axes = plot_cor(cor_res, pbmc_meta, data_to_plot=list(cor_res.columns[:2]), cluster_col="classified")
    assert len(axes) == 2
    for ax in axes:
        plt.close(ax.figure)


def test_plot_best_call(cor_res, pbmc_meta):
    axes = plot_best_call(cor_res, pbmc_meta, cluster_col="classified", plot_r=True)
    assert len(axes) == 2
    for ax in axes:
        plt.close(ax.figure)


def test_plot_best_call_rejects_column_clash(cor_res, pbmc_meta):
    bad_meta = pbmc_meta.copy()
    bad_meta["type"] = "x"
    with pytest.raises(ValueError):
        plot_best_call(cor_res, bad_meta, cluster_col="classified")


def test_plot_cor_heatmap(cor_res):
    ax = plot_cor_heatmap(cor_res)
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_plot_gene(pbmc_matrix_small, pbmc_meta):
    axes = plot_gene(pbmc_matrix_small, pbmc_meta, genes=["PPBP", "LYZ", "NOT_A_GENE"])
    assert len(axes) == 2
    for ax in axes:
        plt.close(ax.figure)


def test_plot_gene_raises_if_no_genes_found(pbmc_matrix_small, pbmc_meta):
    with pytest.raises(ValueError):
        plot_gene(pbmc_matrix_small, pbmc_meta, genes=["NOT_A_GENE"])
