import numpy as np
import pytest

from pyclustifyr.classify import cor_to_call, cor_to_call_rank, cor_to_call_topn
from pyclustifyr.clustify import clustify
from tests.conftest import EXPECTED_DIR, read_expected


@pytest.mark.parametrize("compute_method", ["spearman", "pearson", "cosine", "kendall"])
def test_clustify_matches_r(pbmc_matrix_small, pbmc_meta, cbmc_ref, compute_method):
    res = clustify(
        pbmc_matrix_small,
        cbmc_ref,
        metadata=pbmc_meta,
        cluster_col="classified",
        compute_method=compute_method,
        verbose=False,
    )
    expected = read_expected(f"clustify_{compute_method}")[res.columns]
    diff = np.abs(res.loc[expected.index].to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_clustify_per_cell_matches_r(pbmc_matrix_small, pbmc_meta, cbmc_ref):
    cells = list(pbmc_matrix_small.columns[:50])
    res = clustify(
        pbmc_matrix_small[cells],
        cbmc_ref,
        metadata=pbmc_meta.loc[cells],
        cluster_col="classified",
        per_cell=True,
        verbose=False,
    )
    expected = read_expected("clustify_percell")[res.columns]
    expected.index = expected.index.astype(str)
    diff = np.abs(res.loc[expected.index].to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_clustify_vec_out_matches_r(pbmc_matrix_small, pbmc_meta, cbmc_ref):
    calls = clustify(
        pbmc_matrix_small,
        cbmc_ref,
        metadata=pbmc_meta,
        cluster_col="classified",
        vec_out=True,
        verbose=False,
    )
    expected = (EXPECTED_DIR / "clustify_vec_out.txt").read_text().splitlines()
    assert list(calls) == expected


def test_clustify_requires_metadata_without_per_cell(pbmc_matrix_small, cbmc_ref):
    with pytest.raises(ValueError):
        clustify(pbmc_matrix_small, cbmc_ref, verbose=False)


def test_clustify_rejects_unknown_compute_method(pbmc_matrix_small, pbmc_meta, cbmc_ref):
    with pytest.raises(ValueError):
        clustify(
            pbmc_matrix_small,
            cbmc_ref,
            metadata=pbmc_meta,
            cluster_col="classified",
            compute_method="not_a_method",
            verbose=False,
        )


def test_cor_to_call_matches_r(pbmc_matrix_small, pbmc_meta, cbmc_ref):
    res = clustify(pbmc_matrix_small, cbmc_ref, metadata=pbmc_meta, cluster_col="classified", verbose=False)
    calls = cor_to_call(res)
    expected = read_expected("cor_to_call", index_col=None)
    merged = calls.merge(expected, on="cluster", suffixes=("_py", "_r"))
    assert len(merged) == len(calls) == len(expected)
    assert (merged["type_py"] == merged["type_r"]).all()
    assert np.allclose(merged["r_py"], merged["r_r"], atol=1e-9)


def test_cor_to_call_threshold_matches_r(pbmc_matrix_small, pbmc_meta, cbmc_ref):
    res = clustify(pbmc_matrix_small, cbmc_ref, metadata=pbmc_meta, cluster_col="classified", verbose=False)
    calls = cor_to_call(res, carry_r=True, threshold=0.85)
    expected = read_expected("cor_to_call_threshold", index_col=None)
    merged = calls.merge(expected, on="cluster", suffixes=("_py", "_r"))
    assert (merged["type_py"] == merged["type_r"]).all()


def test_cor_to_call_rank_matches_r(pbmc_matrix_small, pbmc_meta, cbmc_ref):
    res = clustify(pbmc_matrix_small, cbmc_ref, metadata=pbmc_meta, cluster_col="classified", verbose=False)
    ranks = cor_to_call_rank(res, threshold="auto")
    expected = read_expected("cor_to_call_rank", index_col=None)
    merged = ranks.merge(expected, on=["cluster", "type"], suffixes=("_py", "_r"))
    assert len(merged) == len(ranks) == len(expected)
    assert (merged["rank_py"] == merged["rank_r"]).all()


def test_cor_to_call_topn_matches_r(pbmc_matrix_small, pbmc_meta, cbmc_ref):
    res = clustify(pbmc_matrix_small, cbmc_ref, metadata=pbmc_meta, cluster_col="classified", verbose=False)
    topn = cor_to_call_topn(res, threshold=0.5, topn=3)
    expected = read_expected("cor_to_call_topn", index_col=None)
    key_py = set(zip(topn["cluster"], topn["type"]))
    key_r = set(zip(expected["cluster"], expected["type"]))
    assert key_py == key_r
