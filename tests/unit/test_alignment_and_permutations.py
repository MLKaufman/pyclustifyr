"""Classification must respect cell identity and count tied null scores."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyclustifyr.clustify import clustify, clustify_lists
from pyclustifyr.clusters import average_clusters
from pyclustifyr.similarity import permute_similarity


@pytest.fixture
def data():
    expr = pd.DataFrame({"c1": [3., 2., 0., 0.], "c2": [0., 0., 2., 3.]}, index=list("abcd"))
    reference = expr.set_axis(["A", "B"], axis=1)
    meta = pd.DataFrame({"cluster": ["0", "1"]}, index=expr.columns)
    return expr, reference, meta


@pytest.mark.parametrize("series", [False, True])
@pytest.mark.parametrize("n_perm", [0, 3])
def test_metadata_reordering_preserves_scores_and_cell_calls(data, series, n_perm):
    expr, ref, meta = data
    original = meta["cluster"] if series else meta
    reordered = original.iloc[::-1]
    args = dict(cluster_col="cluster", n_perm=n_perm, verbose=False)
    expected = clustify(expr, ref, metadata=original, **args)
    assert_frame_equal(clustify(expr, ref, metadata=reordered, **args), expected)
    calls = clustify(expr, ref, metadata=reordered, vec_out=True, **args)
    assert dict(zip(reordered.index, calls)) == {"c1": "A", "c2": "B"}


@pytest.mark.parametrize("metric", ["hyper", "pct", "posneg", "consensus"])
def test_marker_classification_also_aligns_metadata(data, metric):
    expr, _, meta = data
    markers = pd.DataFrame({"A": ["a", "b"], "B": ["c", "d"]})
    args = dict(cluster_col="cluster", metric=metric, verbose=False)
    expected = clustify_lists(expr, markers, metadata=meta, **args)
    actual = clustify_lists(expr, markers, metadata=meta.iloc[::-1], **args)
    if metric == "pct":
        actual = actual.reindex(expected.index)
    assert_frame_equal(actual, expected)
    calls = clustify_lists(expr, markers, metadata=meta.iloc[::-1], vec_out=True, **args)
    assert calls == ["B", "A"]


@pytest.mark.parametrize("index", [["c1", "unknown"], ["c1", "c1"]])
def test_ambiguous_cell_ids_are_rejected(data, index):
    expr, ref, meta = data
    meta.index = index
    with pytest.raises(ValueError, match="cell IDs"):
        clustify(expr, ref, metadata=meta, cluster_col="cluster", verbose=False)


def test_explicit_cell_column_takes_precedence(data):
    expr, _, meta = data
    shuffled = meta.iloc[::-1].assign(cell=["c2", "c1"])
    shuffled.index = ["row1", "row2"]
    expected = average_clusters(expr, meta)
    assert_frame_equal(average_clusters(expr, shuffled, cell_col="cell"), expected)


def test_default_range_metadata_remains_positional(data):
    expr, ref, meta = data
    expected = clustify(expr, ref, metadata=meta, cluster_col="cluster", verbose=False)
    actual = clustify(expr, ref, metadata=meta.reset_index(drop=True), cluster_col="cluster", verbose=False)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("per_cell", [False, True])
def test_identical_profiles_have_unit_permutation_pvalues(per_cell):
    expr = pd.DataFrame({f"c{i}": [1., 2., 3.] for i in range(4)})
    labels = expr.columns if per_cell else ["A", "A", "B", "B"]
    result = permute_similarity(expr, expr[["c0"]], labels, n_perm=10, per_cell=per_cell,
                                rng=np.random.default_rng(2))
    np.testing.assert_allclose(result["p_val"], 1)


def test_mixed_ties_and_losses_use_corrected_upper_tail(data):
    expr, ref, _ = data

    class IdentityThenSwap:
        def __init__(self):
            self.n = 0

        def permutation(self, labels):
            self.n += 1
            return labels if self.n == 1 else labels[::-1]

    result = permute_similarity(expr, ref, expr.columns, n_perm=2, per_cell=True, rng=IdentityThenSwap())
    np.testing.assert_allclose(result["p_val"], [[2 / 3, 1], [1, 2 / 3]])


def test_undefined_correlations_have_missing_permutation_pvalues(data):
    expr, _, _ = data
    ref = pd.DataFrame({"constant": [1., 1., 1., 1.]}, index=expr.index)
    result = permute_similarity(expr, ref, expr.columns, n_perm=2, per_cell=True)
    assert result["p_val"].isna().all().all()


@pytest.mark.parametrize("n_perm", [0, -1, 1.5])
def test_invalid_permutation_counts_rejected(data, n_perm):
    expr, ref, _ = data
    with pytest.raises(ValueError, match="positive integer"):
        permute_similarity(expr, ref, expr.columns, n_perm=n_perm, per_cell=True)
