"""Numeric representation, missing-score, and file-format regressions."""

import gzip
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyclustifyr.anndata_io import clustify_adata
from pyclustifyr.cellbrowsers import get_ucsc_reference
from pyclustifyr.classify import cor_to_call
from pyclustifyr.clusters import percent_clusters
from pyclustifyr.genelist import binarize_expr
from pyclustifyr.gsea import fgsea_simple, gmt_to_list


@pytest.mark.parametrize("dtype", ["uint8", "uint32", "uint64", "int64"])
def test_gsea_numeric_dtypes_match_float(dtype):
    values = pd.Series([0, 1, 2, 3], index=["zero", "one", "two", "three"], dtype=dtype)
    actual = fgsea_simple({"P": ["zero"]}, values, n_perm=100, rng=np.random.default_rng(1))
    expected = fgsea_simple({"P": ["zero"]}, values.astype(float), n_perm=100, rng=np.random.default_rng(1))
    assert_frame_equal(actual, expected)
    assert actual.iloc[0]["es"] == -1


@pytest.mark.parametrize("ids", [("c1", "c2"), ("001", "002")])
@pytest.mark.parametrize("explicit_id", [False, True])
def test_cellbrowser_aligns_metadata_by_id(monkeypatch, ids, explicit_id):
    c1, c2 = ids
    id_col = "barcode" if explicit_id else "Cell"
    metadata = f"{id_col}\tcluster\n{c2}\tB\n{c1}\tA\n"
    matrix = f"gene\t{c1}\t{c2}\ng1\t10\t1\n"

    def get(url, **kwargs):
        if url.endswith("meta.tsv"):
            return SimpleNamespace(status_code=200, text=metadata)
        return SimpleNamespace(status_code=200, content=gzip.compress(matrix.encode()))

    monkeypatch.setattr("pyclustifyr.cellbrowsers.requests.get", get)
    args = {"cell_col": id_col} if explicit_id else {}
    result = get_ucsc_reference("https://example.test/?ds=test", "cluster", if_log=False, output_log=False, **args)
    assert result.loc["g1"].to_dict() == {"A": 10., "B": 1.}


@pytest.mark.parametrize("metadata", ["Cell\tcluster\nc1\tA\nc1\tB\n", "Cell\tcluster\nc1\tA\nc3\tB\n"])
def test_cellbrowser_rejects_ambiguous_or_mismatched_ids(monkeypatch, metadata):
    def get(url, **kwargs):
        if url.endswith("meta.tsv"):
            return SimpleNamespace(status_code=200, text=metadata)
        return SimpleNamespace(status_code=200, content=gzip.compress(b"gene\tc1\tc2\ng1\t10\t1\n"))
    monkeypatch.setattr("pyclustifyr.cellbrowsers.requests.get", get)
    with pytest.raises(ValueError, match="cell IDs"):
        get_ucsc_reference("https://example.test/?ds=test", "cluster", if_log=False)


@pytest.mark.parametrize("threshold,expected", [(0, "unassigned"), (-1, "valid"), ("auto", "unassigned")])
def test_undefined_correlations_never_win(threshold, expected):
    scores = pd.DataFrame({"constant": [np.nan], "valid": [-.8]}, index=["c1"])
    result = cor_to_call(scores, threshold=threshold)
    assert result.iloc[0]["type"] == expected
    assert result.iloc[0]["r"] == -.8


@pytest.mark.parametrize("threshold", [0, "auto", -1])
def test_all_missing_correlations_are_unassigned(threshold):
    scores = pd.DataFrame(np.nan, index=["c1", "c2"], columns=["A", "B"])
    result = cor_to_call(scores, threshold=threshold, rename_prefix="pred")
    assert len(result) == 2
    assert result["pred_type"].tolist() == ["unassigned", "unassigned"]
    assert result["pred_r"].isna().all()


def test_invalid_reference_is_unassigned_through_anndata():
    expr = pd.DataFrame({"cell": [1., 2., 3.]}, index=["g1", "g2", "g3"])
    reference = pd.DataFrame({"constant": [1., 1., 1.], "opposite": [3., 2., 1.]}, index=expr.index)
    result = clustify_adata(ad.AnnData(expr.T), reference, per_cell=True, threshold=0, verbose=False)
    assert result.obs["type"].tolist() == ["unassigned"]
    assert result.obs["r"].iloc[0] == pytest.approx(-1.)


@pytest.mark.parametrize("cut", [.5, 2., -1.])
def test_percent_clusters_returns_unlogged_fractions(cut):
    expr = pd.DataFrame({"c1": [cut + 1, cut - 1], "c2": [cut + 1, cut]}, index=["all", "half"])
    result = percent_clusters(expr, ["A", "A"], cut_num=cut)
    assert result["A"].tolist() == [1., .5]


def test_percent_clusters_excludes_missing_expression():
    expr = pd.DataFrame({"c1": [3.], "c2": [np.nan]})
    assert percent_clusters(expr, ["A", "A"], cut_num=2).iloc[0, 0] == 1


@pytest.mark.parametrize("cut", [-.5, -10., 0.])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_cutoff_cannot_reselect_excluded_genes(cut, n):
    result = binarize_expr(pd.DataFrame({"c": [3., 2., 1.]}), n=n, cut=cut)
    assert result["c"].tolist() == [1] * n + [0] * (3 - n)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("description", ["na", "https://example.org/pathway", "", "plain description"])
def test_gmt_standard_descriptions(tmp_path, compressed, description):
    path = tmp_path / ("genes.gmt.gz" if compressed else "genes.gmt")
    text = f"# comment\n\nREACTOME_TEST\t{description}\tG1\tG2\r\nSMALL\tna\tG3\n"
    path.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
    assert gmt_to_list(path, cutoff=2) == {"TEST": ["G1", "G2"]}


def test_gmt_custom_separator_remains_supported(tmp_path):
    path = tmp_path / "custom.gmt"
    path.write_text("PATH::G1\tG2\n")
    assert gmt_to_list(path, sep="::") == {"PATH": ["G1", "G2"]}


def test_gmt_malformed_row_reports_line(tmp_path):
    path = tmp_path / "bad.gmt"
    path.write_text("# comment\nPATH\tna\n")
    with pytest.raises(ValueError, match="line 2"):
        gmt_to_list(path)
