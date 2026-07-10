from pathlib import Path

import anndata as ad
import pandas as pd
import pytest
import scipy.io

DATA_DIR = Path(__file__).parent / "data"
EXPECTED_DIR = DATA_DIR / "expected"


def read_expected(name: str, index_col=0) -> pd.DataFrame:
    return pd.read_csv(EXPECTED_DIR / f"{name}.csv", index_col=index_col)


def _load_sparse(name: str):
    mat = scipy.io.mmread(DATA_DIR / f"{name}.mtx.gz").tocsc()
    rownames = (DATA_DIR / f"{name}_rownames.txt").read_text().splitlines()
    colnames = (DATA_DIR / f"{name}_colnames.txt").read_text().splitlines()
    return mat, rownames, colnames


@pytest.fixture(scope="session")
def pbmc_matrix_small():
    mat, rownames, colnames = _load_sparse("pbmc_matrix_small")
    return pd.DataFrame(mat.toarray(), index=rownames, columns=colnames)


@pytest.fixture(scope="session")
def cbmc_ref():
    return pd.read_csv(DATA_DIR / "cbmc_ref.csv", index_col=0)


@pytest.fixture(scope="session")
def cbmc_m():
    return pd.read_csv(DATA_DIR / "cbmc_m.csv", index_col=0)


@pytest.fixture(scope="session")
def pbmc_meta():
    return pd.read_csv(DATA_DIR / "pbmc_meta.csv", index_col=0)


@pytest.fixture(scope="session")
def pbmc_markers():
    df = pd.read_csv(DATA_DIR / "pbmc_markers.csv")
    df["cluster"] = pd.Categorical(df["cluster"].astype(str), categories=[str(i) for i in range(9)])
    return df


@pytest.fixture(scope="session")
def pbmc_markers_m3drop():
    return pd.read_csv(DATA_DIR / "pbmc_markers_M3Drop.csv")


@pytest.fixture(scope="session")
def pbmc_vargenes():
    return (DATA_DIR / "pbmc_vargenes.txt").read_text().splitlines()


@pytest.fixture(scope="session")
def pbmc_anndata(pbmc_matrix_small, pbmc_meta, pbmc_vargenes):
    """Analog of clustifyr's so_pbmc() / sce_pbmc() example objects."""
    meta = pbmc_meta.copy()
    meta["seurat_clusters"] = meta["seurat_clusters"].astype(str)
    adata = ad.AnnData(
        X=pbmc_matrix_small.T.to_numpy(),
        obs=meta,
        var=pd.DataFrame(index=pbmc_matrix_small.index),
    )
    adata.var["highly_variable"] = adata.var_names.isin(pbmc_vargenes)
    return adata
