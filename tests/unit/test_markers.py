import numpy as np
import pandas as pd

from pyclustifyr.markers import (
    append_genes,
    calc_distance,
    check_raw_counts,
    feature_select_pca,
    gene_pct_markerm,
    make_comb_ref,
    pos_neg_marker,
    pos_neg_select,
    ref_feature_select,
    ref_marker_select,
    reverse_marker_matrix,
)
from tests.conftest import EXPECTED_DIR, read_expected


def test_ref_marker_select_cluster_and_ratio_match_r(cbmc_ref):
    res = ref_marker_select(cbmc_ref, cut=2)
    expected = read_expected("ref_marker_select", index_col=None)
    assert res.shape == expected.shape
    assert (res["cluster"].to_numpy() == expected["cluster"].to_numpy()).all()
    assert np.allclose(res["ratio"].to_numpy(), expected["ratio"].to_numpy())


def test_pos_neg_marker_matches_r(cbmc_m):
    res = pos_neg_marker(cbmc_m)
    expected = read_expected("pos_neg_marker").loc[res.index, res.columns]
    assert (res.to_numpy() == expected.to_numpy()).all()


def test_reverse_marker_matrix_matches_r(cbmc_m):
    res = reverse_marker_matrix(cbmc_m)
    expected = read_expected("reverse_marker_matrix", index_col=None)
    assert res.shape == expected.shape
    for col in res.columns:
        assert set(res[col].dropna()) == set(expected[col].dropna())


def test_ref_feature_select_matches_r(cbmc_ref):
    res = ref_feature_select(cbmc_ref, n=50, mode="var")
    expected = (EXPECTED_DIR / "ref_feature_select.txt").read_text().splitlines()
    assert res == expected


def test_feature_select_pca_matches_r(cbmc_ref):
    res = feature_select_pca(cbmc_ref, if_log=False)
    expected = (EXPECTED_DIR / "feature_select_pca.txt").read_text().splitlines()
    assert len(res) == len(expected)
    assert sorted(res) == sorted(expected)


def test_make_comb_ref_matches_r(cbmc_ref):
    res = make_comb_ref(cbmc_ref, sep="_+_")
    expected = read_expected("make_comb_ref")[res.columns]
    diff = np.abs(res.loc[expected.index].to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_append_genes_matches_r(cbmc_ref):
    res = append_genes(["PPBP", "NOTAGENE123"], cbmc_ref)
    expected = read_expected("append_genes")
    assert np.allclose(res.to_numpy(), expected.loc[res.index, res.columns].to_numpy())
    assert (res.loc["NOTAGENE123"] == 0).all()


def test_check_raw_counts_matches_r(pbmc_matrix_small):
    assert check_raw_counts(pbmc_matrix_small) == "log-normalized"


def test_gene_pct_markerm_matches_r(pbmc_matrix_small, cbmc_m, pbmc_meta):
    res = gene_pct_markerm(pbmc_matrix_small, cbmc_m, pbmc_meta, cluster_col="classified")
    expected = read_expected("gene_pct_markerm").loc[res.index, res.columns]
    diff = np.abs(res.to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_pos_neg_select_matches_r(pbmc_matrix_small, pbmc_meta):
    pn_ref = pd.DataFrame({"Myeloid": [1, 0.01, 0]}, index=["CD74", "clustifyr0", "CD79A"])
    res = pos_neg_select(pbmc_matrix_small, pn_ref, pbmc_meta, cluster_col="classified", cutoff_score=0.8)
    expected = read_expected("pos_neg_select").loc[res.index, res.columns]
    diff = np.abs(res.to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_calc_distance_matches_r():
    coords = pd.read_csv(EXPECTED_DIR / "spatial_coords.csv", index_col=0)
    groups = (EXPECTED_DIR / "group_ids.txt").read_text().splitlines()
    res = calc_distance(coords, groups)
    expected = read_expected("calc_distance").loc[res.index, res.columns]
    diff = np.abs(res.to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9
