"""Regression coverage for classification and option-handling bugs."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, pearsonr, spearmanr

from pyclustifyr.anndata_io import clustify_adata
from pyclustifyr.classify import call_consensus, cor_to_call, cor_to_call_rank
from pyclustifyr.clustify import clustify, clustify_lists
from pyclustifyr.markers import ref_feature_select
from pyclustifyr.similarity import calc_similarity, permute_similarity


@pytest.fixture
def small_query():
    expr = pd.DataFrame(
        {"c1": [1., 2., 3., 4.], "c2": [4., 3., 2., 1.]},
        index=["g1", "g2", "g3", "g4"],
    )
    ref = expr.rename(columns={"c1": "A", "c2": "B"})
    meta = pd.DataFrame({"cluster": ["0", "1"]}, index=expr.columns)
    return expr, ref, meta


def test_consensus_uses_ranks_not_scores():
    scores = pd.DataFrame({"A": [.9], "B": [.1]}, index=["0"])
    ranks = cor_to_call_rank(scores)
    result = call_consensus([ranks, ranks])
    assert result.to_dict("records") == [{"cluster": "0", "type": "A", "rank": 1.0}]


def test_consensus_preserves_rank_ties():
    first = cor_to_call_rank(pd.DataFrame({"A": [.9], "B": [.1]}, index=["0"]))
    second = cor_to_call_rank(pd.DataFrame({"A": [.2], "B": [.7]}, index=["0"]))
    result = call_consensus([first, second])
    assert result.iloc[0]["type"] == "A__B"
    assert result.iloc[0]["rank"] == 1.5


@pytest.mark.parametrize("cluster_col", [None, "cluster"])
@pytest.mark.parametrize("vec_out", [False, True])
def test_anndata_per_cell_output(small_query, cluster_col, vec_out):
    expr, ref, meta = small_query
    # Winning types order differs from the cell order.
    expr, meta = expr.iloc[:, ::-1], meta.iloc[::-1]
    original = ad.AnnData(expr.T, obs=meta)
    result = clustify_adata(
        original, ref, cluster_col=cluster_col, per_cell=True,
        vec_out=vec_out, rename_prefix="pred", verbose=False,
    )
    if vec_out:
        assert result == ["B", "A"]
    else:
        assert result.obs.index.equals(original.obs.index)
        assert result.obs["pred_type"].tolist() == ["B", "A"]
        np.testing.assert_allclose(result.obs["pred_r"], 1)
    assert "pred_type" not in original.obs


def test_collapse_aligns_cell_ids_after_call_reordering():
    scores = pd.DataFrame({"A": [.1, .9], "B": [.9, .1]}, index=["c1", "c2"])
    meta = pd.DataFrame({"cluster": ["0", "1"]}, index=["c1", "c2"])
    result = cor_to_call(scores, metadata=meta, collapse_to_cluster=True)
    assert result.set_index("cluster")["type"].to_dict() == {"0": "B", "1": "A"}


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_per_cell_permutation_null_changes_values(small_query, method):
    expr, ref, _ = small_query

    class SwapEveryTime:
        def permutation(self, labels):
            return labels[::-1]

    result = permute_similarity(
        expr, ref, expr.columns, n_perm=5, per_cell=True,
        compute_method=method, rng=SwapEveryTime(),
    )
    np.testing.assert_allclose(result["score"], [[1, -1], [-1, 1]])
    np.testing.assert_allclose(result["p_val"], [[1 / 6, 1], [1, 1 / 6]])


@pytest.mark.parametrize("method,stat", [("pearson", pearsonr), ("spearman", spearmanr), ("kendall", kendalltau)])
def test_rm0_uses_requested_metric_and_pairwise_missing_values(method, stat):
    query = pd.DataFrame({"q": [0., 1., 2., 4., 3., 9.]})
    reference = pd.DataFrame({"r": [8., 1., 4., 2., 3., np.nan]})
    result = calc_similarity(query, reference, method, rm0=True)
    expected = stat([1, 2, 4, 3], [1, 4, 2, 3]).statistic
    assert result.iloc[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize("method", ["cosine", "kl_divergence"])
def test_rm0_rejects_unsupported_metrics(small_query, method):
    expr, ref, _ = small_query
    with pytest.raises(ValueError, match="rm0 is supported only"):
        calc_similarity(expr, ref, method, rm0=True)


@pytest.mark.parametrize("metadata_factory", [list, pd.Series])
def test_vector_metadata_returns_calls_in_cell_order(small_query, metadata_factory):
    expr, ref, _ = small_query
    # Two cells in the same cluster ensure output remains per cell.
    expr = expr[["c2", "c1", "c1"]].set_axis(["b", "a1", "a2"], axis=1)
    calls = clustify(
        expr, ref, metadata=metadata_factory(["1", "0", "0"]),
        vec_out=True, rename_prefix="pred", verbose=False,
    )
    assert calls == ["B", "A", "A"]


def test_per_cell_vector_without_metadata(small_query):
    expr, ref, _ = small_query
    assert clustify(expr, ref, per_cell=True, vec_out=True, verbose=False) == ["A", "B"]


def test_details_output_with_default_verbosity(small_query, capsys):
    expr, _, meta = small_query
    markers = pd.DataFrame({"A": ["g1", "g2"], "B": ["g3", "g4"]})
    result = clustify_lists(
        expr, markers, metadata=meta, cluster_col="cluster", details_out=True,
    )
    assert set(result) == {"res", "details"}
    assert result["res"].shape == (2, 2)
    assert result["details"].loc["0", "A"] == "g1,g2"
    assert "matrix of 2 x 2" in capsys.readouterr().out


def test_details_and_vector_output_are_mutually_exclusive(small_query):
    expr, _, meta = small_query
    with pytest.raises(ValueError, match="details_out and vec_out"):
        clustify_lists(expr, pd.DataFrame(), metadata=meta, details_out=True, vec_out=True)


def test_feature_selection_without_filtering_ranks_all_genes():
    mat = pd.DataFrame([[1, 1, 1], [0, 10, 20], [1, 2, 3], [0, 4, 8]],
                       index=["constant", "high", "low", "medium"])
    assert ref_feature_select(mat, n=3, rm_lowvar=False) == ["high", "medium", "low"]
    assert ref_feature_select(mat, n=4, rm_lowvar=True) == ["high", "medium"]
