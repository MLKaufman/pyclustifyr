import numpy as np
import pytest

from pyclustifyr.clustify import clustify_lists
from tests.conftest import read_expected


@pytest.mark.parametrize("metric", ["hyper", "jaccard", "pct"])
def test_clustify_lists_matches_r(pbmc_matrix_small, pbmc_meta, cbmc_m, metric):
    res = clustify_lists(
        pbmc_matrix_small, cbmc_m, metadata=pbmc_meta, cluster_col="classified", metric=metric, verbose=False
    )
    expected = read_expected(f"clustify_lists_{metric}")[res.columns]
    diff = np.abs(res.loc[expected.index].to_numpy() - expected.to_numpy()).max()
    assert diff < 1e-9
