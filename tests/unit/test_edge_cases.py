"""Boundary and option-interaction tests complementing R fixture parity."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyclustifyr.classify import call_consensus, cor_to_call_rank
from pyclustifyr.clusters import average_clusters
from pyclustifyr.clustify import clustify, clustify_lists
from pyclustifyr.genelist import _p_adjust_holm, binarize_expr, compare_lists, matrixize_markers
from pyclustifyr.markers import feature_select_pca
from pyclustifyr.plot import plot_best_call


@pytest.fixture
def data():
    expr = pd.DataFrame({"b": [0., 0., 4., 3.], "a1": [4., 3., 0., 0.], "a2": [3., 4., 0., 0.]},
                        index=["g1", "g2", "g3", "g4"])
    meta = pd.DataFrame({"cluster": ["b", "a", "a"], "UMAP_1": [1., 2., 3.], "UMAP_2": [4., 5., 6.]},
                        index=expr.columns)
    markers = pd.DataFrame({"A": ["g1", "g2"], "B": ["g3", "g4"]})
    return expr, meta, markers


@pytest.mark.parametrize("n", [0, -1, 1, 1.5, np.nan])
def test_invalid_gene_universe_is_rejected(data, n):
    expr, _, markers = data
    with pytest.raises(ValueError, match="universe"):
        compare_lists((expr > 0).astype(int), markers, n=n)


def test_holm_keeps_missing_pvalues():
    result = _p_adjust_holm([.01, np.nan, .2])
    np.testing.assert_allclose(result, [.03, np.nan, .4], equal_nan=True)


@pytest.mark.parametrize("output_high", [False, True])
def test_absent_and_single_gene_marker_sets_cannot_win(data, output_high):
    expr, meta, _ = data
    markers = pd.DataFrame({"valid": ["g2", "g1"], "absent": ["x", "y"], "single": ["g1", "x"]})
    scores = clustify_lists(expr, markers, metadata=meta, cluster_col="cluster", metric="spearman",
                            output_high=output_high, verbose=False)
    assert scores[["absent", "single"]].isna().all().all()
    calls = clustify_lists(expr, markers, metadata=meta, cluster_col="cluster", metric="spearman",
                           output_high=output_high, vec_out=True, verbose=False)
    assert calls == ["valid"] * 3
    empty = clustify_lists(expr, markers[["absent"]], metadata=meta, cluster_col="cluster",
                           metric="spearman", output_high=output_high, vec_out=True, verbose=False)
    assert empty == ["unassigned"] * 3


@pytest.mark.parametrize("delta", [-1, 1])
def test_metadata_length_is_validated_for_dataframes(data, delta):
    expr, meta, _ = data
    meta = meta.iloc[:2] if delta < 0 else pd.concat([meta, meta.iloc[:1]])
    with pytest.raises(ValueError, match="number of columns"):
        average_clusters(expr, meta)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
@pytest.mark.parametrize("ranked", [False, True])
def test_marker_conversion_drops_missing_values(missing, ranked):
    wide = pd.DataFrame({"A": ["g1", missing], "B": ["g2", "g3"]})
    long = pd.DataFrame({"cluster": ["A", "B", "B"], "gene": ["g1", "g2", "g3"]})
    assert_frame_equal(matrixize_markers(wide, ranked=ranked), matrixize_markers(long, ranked=ranked))


def test_all_missing_markers_return_empty_matrix():
    result = matrixize_markers(pd.DataFrame({"A": [None], "B": [np.nan]}))
    assert result.shape == (0, 2)


@pytest.mark.parametrize("cut", [0, .5, 2])
def test_binarization_boundary_is_strict_and_binary(cut):
    expr = pd.DataFrame({"c": [cut - 1, cut, cut + 1]})
    result = binarize_expr(expr, cut=cut)
    assert result["c"].tolist() == [0, 0, 1]


@pytest.mark.parametrize("n_perm", [0, 2])
def test_missing_numeric_cluster_labels_work_for_scores_and_vectors(data, n_perm):
    expr, _, _ = data
    ref = expr[["b", "a1"]].set_axis(["B", "A"], axis=1)
    meta = pd.DataFrame({"cluster": [0, np.nan, np.nan]}, index=expr.columns)
    kwargs = dict(metadata=meta, cluster_col="cluster", n_perm=n_perm, verbose=False,
                  rng=np.random.default_rng(1))
    scores = clustify(expr, ref, **kwargs)
    assert set(scores.index) == {0., "orig.NA"}
    assert clustify(expr, ref, vec_out=True, **kwargs) == ["B", "A", "A"]
    assert meta["cluster"].isna().sum() == 2


@pytest.mark.parametrize("options", [{"topn": 0}, {"cut": 100}, {"if_log": False, "topn": 1},
                                     {"genome_n": 5}, {"low_threshold_cell": 2}])
def test_consensus_matches_explicit_component_calls(data, options):
    expr, meta, markers = data
    common = dict(metadata=meta, cluster_col="cluster", verbose=False, **options)
    results = [clustify_lists(expr, markers, metric=metric, **common)
               for metric in ("hyper", "jaccard", "pct", "posneg")]
    expected = call_consensus([cor_to_call_rank(r) for r in results])
    actual = clustify_lists(expr, markers, metric="consensus", **common)
    assert_frame_equal(actual, expected)
    if "low_threshold_cell" in options:
        assert actual["cluster"].tolist() == ["a"]


@pytest.mark.parametrize("collapse", [True, "cluster"])
def test_collapsed_plot_preserves_all_coordinates(data, collapse):
    _, meta, _ = data
    scores = pd.DataFrame({"A": [.1, .9, .7], "B": [.9, .1, .3]}, index=meta.index)
    axes = plot_best_call(scores, meta, per_cell=True, collapse_to_cluster=collapse, plot_r=True)
    for ax in axes:
        coords = np.concatenate([c.get_offsets() for c in ax.collections])
        assert set(map(tuple, coords)) == set(map(tuple, meta[["UMAP_1", "UMAP_2"]].to_numpy()))
        plt.close(ax.figure)
    assert {c.get_label() for c in axes[0].collections} == {"A", "B"}


def test_small_pca_limits_components(data):
    expr, _, _ = data
    assert feature_select_pca(expr) == feature_select_pca(expr, n_pcs=3)
    pcs = pd.DataFrame({"PC1": [0.1, 0.9]}, index=["g1", "g2"])
    assert feature_select_pca(pcs=pcs) == ["g2"]
