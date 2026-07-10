import numpy as np
import pytest

from pyclustifyr.clusters import average_clusters
from tests.conftest import read_expected

METHODS = ["mean", "median", "trimean", "truncate", "min", "max"]


@pytest.mark.parametrize("method", METHODS)
def test_average_clusters_matches_r(pbmc_matrix_small, pbmc_meta, method):
    result = average_clusters(
        pbmc_matrix_small, pbmc_meta, cluster_col="classified", if_log=False, method=method
    )
    expected = read_expected(f"avg_{method}")[result.columns]
    diff = np.abs(result.loc[expected.index].to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_average_clusters_if_log_matches_r(pbmc_matrix_small, pbmc_meta):
    result = average_clusters(pbmc_matrix_small, pbmc_meta, cluster_col="classified", if_log=True)
    expected = read_expected("avg_mean_iflog")[result.columns]
    diff = np.abs(result.loc[expected.index].to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9


def test_average_clusters_vector_metadata(pbmc_matrix_small, pbmc_meta):
    vec_result = average_clusters(pbmc_matrix_small, list(pbmc_meta["classified"]), if_log=False)
    df_result = average_clusters(pbmc_matrix_small, pbmc_meta, cluster_col="classified", if_log=False)
    assert set(vec_result.columns) == set(df_result.columns)
    diff = np.abs(vec_result[df_result.columns].to_numpy() - df_result.to_numpy()).max()
    assert diff < 1e-12


def test_average_clusters_requires_matching_length(pbmc_matrix_small):
    with pytest.raises(ValueError):
        average_clusters(pbmc_matrix_small, ["a", "b", "c"])
