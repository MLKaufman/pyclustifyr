import numpy as np

from pyclustifyr.anndata_io import clustify_adata, object_data
from tests.conftest import read_expected


def test_object_data_matches_source(pbmc_anndata, pbmc_matrix_small, pbmc_meta, pbmc_vargenes):
    expr = object_data(pbmc_anndata, "data")
    assert expr.shape == pbmc_matrix_small.shape
    diff = np.abs(expr.loc[pbmc_matrix_small.index, pbmc_matrix_small.columns].to_numpy() - pbmc_matrix_small.to_numpy()).max()
    assert diff < 1e-9

    var_genes = object_data(pbmc_anndata, "var.genes", n_genes=0)
    assert set(var_genes) == set(pbmc_vargenes)


def test_clustify_adata_cormat_matches_r(pbmc_anndata, cbmc_ref):
    res = clustify_adata(pbmc_anndata, cbmc_ref, cluster_col="seurat_clusters", obj_out=False, verbose=False)
    expected = read_expected("clustify_seurat_cormat")
    expected.index = expected.index.astype(str)
    expected = expected.loc[res.index, res.columns]
    diff = np.abs(res.to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_clustify_adata_obj_out_matches_r(pbmc_anndata, cbmc_ref):
    result = clustify_adata(pbmc_anndata, cbmc_ref, cluster_col="seurat_clusters", obj_out=True, verbose=False)
    expected = read_expected("clustify_seurat_obj_meta")
    expected.index = expected.index.astype(str)

    merged = result.obs[["seurat_clusters", "type", "r"]].merge(
        expected, left_index=True, right_index=True, suffixes=("_py", "_r")
    )
    assert len(merged) == len(expected)
    assert (merged["type_py"] == merged["type_r"]).all()
    assert np.allclose(merged["r_py"], merged["r_r"])

    # original AnnData is untouched (write_meta returns a copy)
    assert "type" not in pbmc_anndata.obs.columns
