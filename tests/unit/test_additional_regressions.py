"""Regression tests for marker output, reclassification, and GSEA edge cases."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyclustifyr.anndata_io import clustify_adata, clustify_lists_adata
from pyclustifyr.clustify import _marker_calls, clustify, clustify_lists
from pyclustifyr.gsea import fgsea_simple, run_gsea
from pyclustifyr.similarity import permute_similarity
from pyclustifyr.clusters import average_clusters
from pyclustifyr.similarity import calc_similarity


@pytest.fixture
def marker_query():
    expr = pd.DataFrame(
        {"b": [0., 0., 8., 10.], "a1": [10., 8., 0., 0.], "a2": [10., 8., 0., 0.]},
        index=["g1", "g2", "g3", "g4"],
    )
    ref = expr[["a1", "b"]].set_axis(["A", "B"], axis=1)
    meta = pd.DataFrame({"cluster": ["1", "0", "0"]}, index=expr.columns)
    markers = pd.DataFrame({"A": ["g1", "g2"], "B": ["g3", "g4"]})
    return expr, ref, meta, markers


@pytest.mark.parametrize("metric", ["hyper", "spearman"])
def test_raw_marker_scores_and_calls_agree(marker_query, metric):
    expr, _, meta, markers = marker_query
    # Opposite ordered marker lists distinguish the rank-distance metric.
    if metric == "spearman":
        markers = pd.DataFrame({"A": list(expr.index), "B": list(expr.index[::-1])})
    args = dict(metadata=meta, cluster_col="cluster", metric=metric, verbose=False)
    high = clustify_lists(expr, markers, **args, vec_out=True)
    low = clustify_lists(expr, markers, **args, output_high=False, vec_out=True)
    assert low == high == ["B", "A", "A"]
    result = clustify_lists_adata(
        ad.AnnData(expr.T, obs=meta), markers, cluster_col="cluster", metric=metric,
        output_high=False, verbose=False,
    )
    assert result.obs["type"].tolist() == high
    if metric == "hyper":
        raw = clustify_lists(expr, markers, **args, output_high=False)
        assert raw.loc["0", "A"] < 1e-6
        assert raw.loc["0", "B"] == 1


@pytest.mark.parametrize("per_cell", [False, True])
@pytest.mark.parametrize("prefix", [None, "pred"])
def test_reclassification_replaces_old_predictions(marker_query, per_cell, prefix):
    expr, ref, meta, markers = marker_query
    col = f"{prefix}_type" if prefix else "type"
    score_col = f"{prefix}_r" if prefix else "r"
    meta[col] = "OLD"
    meta[score_col] = -100.
    args = dict(cluster_col="cluster", per_cell=per_cell, rename_prefix=prefix, verbose=False)
    calls = clustify(expr, ref, metadata=meta, vec_out=True, **args)
    assert calls == ["B", "A", "A"]
    original = ad.AnnData(expr.T, obs=meta)
    result = clustify_adata(original, ref, **args)
    assert result.obs[col].tolist() == calls
    np.testing.assert_allclose(result.obs[score_col], 1)
    assert original.obs[col].tolist() == ["OLD"] * 3
    assert not any(c.endswith(".clustify") for c in result.obs.columns)
    assert clustify_lists(expr, markers, metadata=meta, vec_out=True, **args) == calls


@pytest.mark.parametrize("metric", ["pct", "posneg", "consensus"])
@pytest.mark.parametrize("per_cell", [False, True])
def test_cluster_marker_outputs_expand_to_cells(marker_query, metric, per_cell):
    expr, _, meta, markers = marker_query
    args = dict(cluster_col="cluster", metric=metric, per_cell=per_cell, verbose=False)
    calls = clustify_lists(expr, markers, metadata=meta, vec_out=True, rename_prefix="pred", **args)
    assert calls == ["B", "A", "A"]
    result = clustify_lists_adata(ad.AnnData(expr.T, obs=meta), markers, rename_prefix="pred", **args)
    assert result.obs.index.equals(meta.index)
    assert result.obs["pred_type"].tolist() == calls
    if metric == "consensus":
        assert "pred_rank" in result.obs
        assert "pred_r" not in result.obs
        raw = clustify_lists_adata(ad.AnnData(expr.T, obs=meta), markers, obj_out=False, **args)
        assert list(raw.columns) == ["cluster", "type", "rank"]


@pytest.mark.parametrize("if_log", [False, True])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_permutations_preserve_preprocessing(if_log, method):
    expr = pd.DataFrame(
        {"c1": [10., 0., 2., 3.], "c2": [0., 8., 2., 1.], "c3": [0., 1., 7., 2.]},
        index=["g1", "g2", "g3", "g4"],
    )
    ref = expr[["c1", "c3"]]
    groups = ["a", "a", "b"]
    kwargs = dict(metadata=groups, if_log=if_log, low_threshold_cell=2,
                  compute_method=method, verbose=False)
    observed = clustify(expr, ref, **kwargs)
    permuted = clustify(expr, ref, n_perm=3, rng=np.random.default_rng(42), **kwargs)
    assert_frame_equal(permuted, observed)
    assert list(permuted.index) == ["a"]

    class Reverse:
        def permutation(self, labels):
            return labels[::-1]

    result = permute_similarity(expr, ref, groups, n_perm=2, rng=Reverse(),
                                if_log=if_log, low_threshold=2, compute_method=method)
    shuffled = average_clusters(expr, groups[::-1], if_log=if_log, low_threshold=2)
    null_score = calc_similarity(shuffled, ref, method)
    assert_frame_equal(result["p_val"], ((null_score >= observed).astype(float) * 2 + 1) / 3)


def test_duplicate_pathway_genes_do_not_change_gsea():
    stats = pd.Series([4., 3., 2., 1.], index=["g1", "g2", "g3", "g4"])
    expected = fgsea_simple({"P": ["g1"]}, stats, n_perm=50, rng=np.random.default_rng(1))
    actual = fgsea_simple({"P": ["g1", "g1"]}, stats, n_perm=50, rng=np.random.default_rng(1))
    assert_frame_equal(actual, expected)
    assert actual.iloc[0]["size"] == 1
    assert abs(actual.iloc[0]["es"]) <= 1


def test_scaled_gsea_excludes_constant_genes_from_entire_null_universe():
    expr = pd.DataFrame({"A": [1., 4., 3., 1.], "B": [1., 1., 2., 3.]},
                        index=["constant", "g2", "g3", "g4"])
    actual = run_gsea(expr, {"P": ["constant", "g2"]}, per_cell=True, scale=True,
                      n_perm=100, rng=np.random.default_rng(1))
    expected = run_gsea(expr.drop(index="constant"), {"P": ["g2"]}, per_cell=True,
                        scale=True, n_perm=100, rng=np.random.default_rng(1))
    assert_frame_equal(actual, expected)
    assert np.isfinite(actual[["pval", "nes"]]).all().all()


def test_gsea_undefined_sets_have_missing_pvalues():
    expr = pd.DataFrame({"A": [1., 4., 3., 1.], "B": [1., 1., 2., 3.]},
                        index=["constant", "g2", "g3", "g4"])
    result = run_gsea(expr, {"empty": ["constant"], "full": ["g2", "g3", "g4"]},
                      per_cell=True, scale=True, n_perm=20, rng=np.random.default_rng(1))
    assert result.shape == (4, 3)
    assert result[["pval", "nes"]].isna().all().all()


def test_gsea_filters_nonfinite_statistics():
    stats = pd.Series([4., np.inf, np.nan, 2., 1.], index=list("abcde"))
    actual = fgsea_simple({"P": list("abc")}, stats, n_perm=30, rng=np.random.default_rng(2))
    expected = fgsea_simple({"P": ["a"]}, stats.loc[list("ade")],
                            n_perm=30, rng=np.random.default_rng(2))
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("metric", ["hyper", "gsea"])
def test_raw_pvalue_calls_use_score_threshold(metric):
    raw = pd.DataFrame({"A": [0.001, 0.5], "B": [1.0, 0.6]}, index=["0", "1"])
    result = _marker_calls(raw, metric, False, "cluster", threshold=2)
    assert result["type"].tolist() == ["A", "unassigned"]
    np.testing.assert_allclose(result["r"], [3, -np.log10(0.5)])
