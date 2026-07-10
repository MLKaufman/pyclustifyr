import numpy as np
import pytest

from pyclustifyr.clusters import average_clusters
from pyclustifyr.genelist import binarize_expr, compare_lists, get_vargenes, matrixize_markers
from tests.conftest import read_expected


@pytest.fixture(scope="module")
def pbmc_avgb(pbmc_matrix_small, pbmc_meta):
    avg = average_clusters(pbmc_matrix_small, pbmc_meta, cluster_col="classified")
    return binarize_expr(avg)


@pytest.fixture(scope="module")
def mm(pbmc_markers):
    return matrixize_markers(pbmc_markers)


def test_binarize_expr_matches_r(pbmc_avgb):
    expected = read_expected("pbmc_avgb")[pbmc_avgb.columns]
    diff = np.abs(pbmc_avgb.loc[expected.index].to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_matrixize_markers_unranked_matches_r(mm):
    expected = read_expected("mm_unranked")
    expected.columns = mm.columns
    assert mm.shape == expected.shape
    assert (mm.astype(str).to_numpy() == expected.astype(str).to_numpy()).all()


def test_matrixize_markers_ranked_matches_r(pbmc_markers):
    ranked = matrixize_markers(pbmc_markers, ranked=True)
    expected = read_expected("mm_ranked")
    expected = expected.loc[ranked.index]
    expected.columns = ranked.columns
    diff = np.abs(ranked.to_numpy() - expected.to_numpy()).max()
    assert diff == 0


def test_get_vargenes_unranked(mm):
    assert len(get_vargenes(mm)) == 666


def test_get_vargenes_ranked(pbmc_markers):
    ranked = matrixize_markers(pbmc_markers, ranked=True)
    assert len(get_vargenes(ranked)) == 666


@pytest.mark.parametrize("metric", ["hyper", "jaccard", "spearman"])
def test_compare_lists_matches_r(pbmc_avgb, mm, metric):
    res = compare_lists(pbmc_avgb, mm, metric=metric)
    expected = read_expected(f"compare_{metric}")
    expected.columns = res.columns
    expected = expected.loc[res.index]
    diff = np.abs(res.to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_compare_lists_with_plain_gene_matrix(pbmc_avgb, cbmc_m):
    res = compare_lists(pbmc_avgb, cbmc_m, metric="hyper")
    expected = read_expected("compare_hyper_cbmc")
    expected = expected.loc[res.index, res.columns]
    diff = np.abs(res.to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_compare_lists_rejects_unknown_metric(pbmc_avgb, mm):
    with pytest.raises(ValueError):
        compare_lists(pbmc_avgb, mm, metric="not_a_metric")
