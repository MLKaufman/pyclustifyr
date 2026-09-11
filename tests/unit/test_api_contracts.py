"""Regression checks for contracts settled before the 1.0 API freeze."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyclustifyr import (
    calc_similarity,
    clustify,
    clustify_adata,
    clustify_lists,
    collapse_to_cluster,
    compare_lists,
    cor_to_call,
    cor_to_call_topn,
    downsample_matrix,
    write_meta,
)
from pyclustifyr.similarity import permute_similarity, vector_similarity


@pytest.fixture
def inputs():
    genes = list("abcdef")
    a, b = [10, 8, 0, 0, 1, 2], [0, 0, 8, 10, 2, 1]
    cells = [f"cell{i}" for i in range(20)]
    expr = pd.DataFrame(np.array([a] * 10 + [b] * 10).T, index=genes, columns=cells)
    ref = pd.DataFrame({"A": a, "B": b}, index=genes)
    meta = pd.DataFrame({"cluster": ["0"] * 10 + ["1"] * 10}, index=cells)
    return expr, ref, meta


def test_write_meta_aligns_and_copies(inputs):
    expr, _, meta = inputs
    obj = ad.AnnData(expr.T, obs=meta.copy())
    reordered = obj.obs.iloc[::-1].copy()
    result = write_meta(obj, reordered)
    assert_frame_equal(result.obs, obj.obs)
    np.testing.assert_array_equal(result.X, obj.X)
    result.obs.iloc[0, 0] = "changed"
    assert obj.obs.iloc[0, 0] == reordered.loc["cell0", "cluster"] == "0"


@pytest.mark.parametrize("kind", ["duplicate", "missing", "extra", "wrong"])
def test_write_meta_rejects_bad_ids(inputs, kind):
    expr, _, meta = inputs
    obj = ad.AnnData(expr.T, obs=meta.copy())
    bad = meta.copy()
    if kind == "duplicate":
        bad.index = ["cell0"] * len(bad)
    elif kind == "missing":
        bad = bad.iloc[:-1]
    elif kind == "extra":
        bad.loc["extra"] = "0"
    else:
        bad.index = ["wrong"] + list(bad.index[1:])
    with pytest.raises(ValueError, match="cell IDs"):
        write_meta(obj, bad)


@pytest.mark.parametrize("metric", ["hyper", "jaccard"])
def test_overlap_metrics_never_infer_rank_distance(metric):
    ranked = pd.DataFrame({"first": [0, 1, 0], "second": [3, 2, 1]}, index=list("abc"))
    markers = pd.DataFrame({"T": ["a", "b"]})
    for mat in (ranked, ranked.iloc[:, ::-1]):
        with pytest.raises(ValueError, match="binary"):
            compare_lists(mat, markers, metric=metric)
    with pytest.raises(ValueError, match="Unknown metric"):
        compare_lists(ranked, markers, metric="typo")
    assert_frame_equal(
        compare_lists(ranked, markers, metric="rank_distance"),
        compare_lists(ranked, markers, metric="spearman"),
    )


def test_rank_distance_alias_in_classification(inputs):
    expr, _, meta = inputs
    markers = pd.DataFrame({"A": ["a", "b"], "B": ["c", "d"]})
    args = dict(
        metadata=meta,
        cluster_col="cluster",
        if_log=False,
        verbose=False,
        vec_out=True,
        output_high=False,
    )
    assert clustify_lists(
        expr, markers, metric="rank_distance", **args
    ) == clustify_lists(expr, markers, metric="spearman", **args)


@pytest.mark.parametrize("carry_r", [False, True])
def test_collapse_filters_numeric_scores_and_retains_rejected_groups(carry_r):
    scores = pd.DataFrame(
        {"A": [0.1, 0.1, 0.9, np.nan, 0.1], "B": [0.0, 0.0, 0.0, np.nan, 0.1]},
        index=list("abcde"),
    )
    meta = pd.DataFrame({"cluster": ["x", "x", "x", "y", "z"]}, index=scores.index)
    result = cor_to_call(
        scores, meta, collapse_to_cluster=True, threshold=0.5, carry_r=carry_r
    ).set_index("cluster")
    assert result.loc["x", "type"] == "A"
    assert result.loc["x", "n"] == 1
    assert result.loc["x", "sum"] == 0.9
    assert result.loc[["y", "z"], "n"].tolist() == [0, 0]
    assert result.loc[["y", "z"], "sum"].isna().all()
    expected = "r<0.5, unassigned" if carry_r else "unassigned"
    assert result.loc[["y", "z"], "type"].tolist() == [expected, expected]


def test_direct_collapse_uses_score_not_type_string():
    res = pd.DataFrame(
        {
            "cell": ["a", "b", "c"],
            "type": ["unassigned", "T", "T"],
            "r": [1.0, 0.1, np.nan],
        }
    )
    meta = pd.DataFrame({"cluster": ["x"] * 3}, index=list("abc"))
    result = collapse_to_cluster(res, meta, "cluster", threshold=0.5)
    # A real type name must not be mistaken for a rejection marker.
    assert result["type"].tolist() == ["unassigned"]
    assert result["n"].tolist() == [1]


@pytest.mark.parametrize("topn", [False, True])
@pytest.mark.parametrize("explicit_ids", [False, True])
def test_collapse_string_selects_real_group(topn, explicit_ids):
    scores = pd.DataFrame({"A": [0.9, 0.1], "B": [0.1, 0.9]}, index=["a", "b"])
    meta = pd.DataFrame(
        {"cluster": ["x", "x"], "other": ["u", "v"]}, index=scores.index
    )
    if explicit_ids:
        meta = meta.rename_axis("cell").reset_index()
    if topn:
        result = cor_to_call_topn(
            scores, meta, col="cell", collapse_to_cluster="other", topn=1
        )
        assert result.set_index("cell")["type"].to_dict() == {"a": "A", "b": "B"}
        assert set(result.type2) == {"u", "v"}
    else:
        result = cor_to_call(
            scores, meta, cluster_col="cell", collapse_to_cluster="other"
        )
        assert result.set_index("other")["type"].to_dict() == {"u": "A", "v": "B"}


@pytest.mark.parametrize("fn", [cor_to_call, cor_to_call_topn])
def test_collapse_true_and_invalid_groups(fn):
    scores = pd.DataFrame({"A": [0.9, 0.8]}, index=["a", "b"])
    meta = pd.DataFrame({"cluster": ["x", "x"]}, index=scores.index)
    result = fn(scores, meta, collapse_to_cluster=True)
    assert set(result.type) == {"A"}
    with pytest.raises(ValueError, match="grouping column"):
        fn(scores, meta, collapse_to_cluster="missing")


@pytest.mark.parametrize(
    "method", ["pearson", "spearman", "kendall", "cosine", "kl_divergence"]
)
def test_similarity_aligns_labels_for_every_metric(method):
    query = pd.DataFrame({"Q": [1.0, 2.0, 4.0]}, index=list("abc"))
    ref = query.rename(columns={"Q": "R"})
    assert_frame_equal(
        calc_similarity(query, ref, method),
        calc_similarity(query, ref.iloc[::-1], method),
    )
    extra = pd.concat([ref, pd.DataFrame({"R": [99.0]}, index=["extra"])])
    assert_frame_equal(
        calc_similarity(query, ref, method), calc_similarity(query, extra, method)
    )
    with pytest.raises(ValueError, match="unique"):
        calc_similarity(query, ref.set_axis(["a", "a", "b"]), method)
    with pytest.raises(ValueError, match="shared genes"):
        calc_similarity(query, ref.set_axis(["x", "y", "z"]), method)


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_rm0_aligns_labels(method):
    query = pd.DataFrame({"Q": [0.0, 2.0, 4.0, 3.0]}, index=list("abcd"))
    ref = pd.DataFrame({"R": [1.0, 3.0, 2.0, 4.0]}, index=list("abcd"))
    assert_frame_equal(
        calc_similarity(query, ref, method, rm0=True),
        calc_similarity(query, ref.iloc[::-1], method, rm0=True),
    )


def test_unused_options_raise_but_kl_options_work(inputs):
    expr, ref, meta = inputs
    args = dict(metadata=meta, cluster_col="cluster", verbose=False)
    with pytest.raises(TypeError, match="compute_methd"):
        clustify(expr, ref, compute_methd="pearson", **args)
    with pytest.raises(TypeError, match="total_reads"):
        clustify(expr, ref, total_reads=100, **args)
    assert clustify(
        expr,
        ref,
        compute_method="kl_divergence",
        if_log=False,
        total_reads=100,
        max_kl=2,
        **args,
    ).shape == (2, 2)
    with pytest.raises(TypeError, match="Unsupported"):
        vector_similarity(np.ones(3), np.ones(3), "cosine", ignored=True)
    markers = pd.DataFrame({"A": ["a", "b"]})
    with pytest.raises(TypeError, match="conversion options"):
        clustify_lists(expr, markers, ranked=True, **args)
    with pytest.raises(TypeError):
        clustify_lists(expr, markers, marker_inmatrix=False, raanked=True, **args)


def test_explicit_permutation_result_and_metadata_alignment(inputs):
    expr, ref, meta = inputs
    args = dict(
        cluster_col="cluster",
        if_log=False,
        verbose=False,
        n_perm=4,
        return_pvalues=True,
    )
    result = clustify(
        expr, ref, metadata=meta.iloc[::-1], rng=np.random.default_rng(42), **args
    )
    expected = permute_similarity(
        expr,
        ref,
        meta.cluster.tolist(),
        n_perm=4,
        if_log=False,
        rng=np.random.default_rng(42),
    )
    assert set(result) == {"score", "p_val"}
    for key in result:
        assert_frame_equal(result[key], expected[key])
    obj = ad.AnnData(expr.T, obs=meta.copy())
    wrapped = clustify_adata(
        obj, ref, obj_out=False, rng=np.random.default_rng(42), **args
    )
    for key in result:
        assert_frame_equal(result[key], wrapped[key])
    with pytest.raises(ValueError, match="obj_out=False"):
        clustify_adata(obj, ref, **args)
    with pytest.warns(FutureWarning, match="discards"):
        legacy = clustify(
            expr, ref, metadata=meta, cluster_col="cluster", n_perm=2, verbose=False
        )
    assert isinstance(legacy, pd.DataFrame)


@pytest.mark.parametrize("args", [{"n_perm": 0}, {"n_perm": 2, "vec_out": True}])
def test_permutation_return_conflicts(inputs, args):
    expr, ref, meta = inputs
    with pytest.raises(ValueError, match="return_pvalues"):
        clustify(
            expr, ref, metadata=meta, cluster_col="cluster", return_pvalues=True, **args
        )


def test_sampling_counts_fractions_alignment_and_seed():
    mat = pd.DataFrame([np.arange(10)], columns=[f"c{i}" for i in range(10)])
    meta = pd.DataFrame({"cluster": ["A"] * 7 + ["B"] * 3}, index=mat.columns)

    def sample(**kwargs):
        return downsample_matrix(
            mat, metadata=meta, rng=np.random.default_rng(42), **kwargs
        )

    result = sample(n=5)
    assert meta.loc[result.columns, "cluster"].value_counts().to_dict() == {
        "A": 4,
        "B": 1,
    }
    assert_frame_equal(
        result,
        downsample_matrix(
            mat, n=5, metadata=meta.iloc[::-1], rng=np.random.default_rng(42)
        ),
    )
    assert_frame_equal(result, sample(frac=0.5))
    assert sample(frac=1).shape[1] == 10
    assert sample(frac=0).shape[1] == 0
    assert sample(n=1).shape[1] == 1
    equal = sample(n=2, per_cluster=True)
    assert meta.loc[equal.columns, "cluster"].value_counts().to_dict() == {
        "A": 2,
        "B": 2,
    }
    assert sample(n=3, keep_cluster_proportions=False).shape[1] == 3
    with pytest.raises(ValueError, match="match"):
        downsample_matrix(mat, n=1, metadata=meta.iloc[:-1])
    with pytest.raises(ValueError, match="unique"):
        downsample_matrix(mat, n=1, metadata=meta.set_axis(["x"] * 10))


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"n": 1, "frac": 0.5},
        {"n": -0.1},
        {"n": 1.5},
        {"n": True},
        {"n": 21},
        {"frac": -1},
        {"frac": 1.1},
        {"frac": float("nan")},
        {"frac": float("inf")},
        {"frac": 0.5, "per_cluster": True},
        {"n": 11, "per_cluster": True},
        {"n": 1, "per_cluster": True, "keep_cluster_proportions": False},
    ],
)
def test_invalid_sampling_requests(inputs, args):
    expr, _, meta = inputs
    with pytest.raises(ValueError):
        downsample_matrix(expr, metadata=meta, **args)


def test_collapse_categorical_groups_do_not_create_zero_count_winners():
    scores = pd.DataFrame({"A": [0.1, 0.2]}, index=["a", "b"])
    meta = pd.DataFrame(
        {"cluster": pd.Categorical(["x", "y"], categories=["x", "y", "unused"])},
        index=scores.index,
    )
    result = cor_to_call(scores, meta, collapse_to_cluster=True, threshold=0.5)
    assert set(result.cluster) == {"x", "y"}
    assert result["n"].tolist() == [0, 0]
    assert result["sum"].isna().all()
