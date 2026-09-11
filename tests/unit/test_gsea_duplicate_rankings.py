"""Duplicate ranking identifiers must not silently select the last score."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyclustifyr.gsea import calculate_pathway_gsea, fgsea_simple, run_gsea


@pytest.mark.parametrize("duplicate_value", [-10., 10., np.nan])
def test_gsea_rejects_duplicate_ids_before_filtering(duplicate_value):
    stats = pd.Series([10., 3., 2., 1., duplicate_value], index=["target", "g2", "g3", "g4", "target"])
    with pytest.raises(ValueError, match="duplicate IDs.*target"):
        fgsea_simple({"P": ["target"]}, stats, n_perm=10)


@pytest.mark.parametrize("entrypoint", ["run_gsea", "calculate_pathway_gsea"])
def test_wrappers_reject_duplicate_rankings(entrypoint):
    matrix = pd.DataFrame({"A": [10., 3., 2., 1., -10.], "B": [-10., 1., 2., 3., 10.]},
                          index=["target", "g2", "g3", "g4", "target"])
    with pytest.raises(ValueError, match="ranking gene IDs must be unique"):
        if entrypoint == "run_gsea":
            run_gsea(matrix, {"P": ["target"]}, per_cell=True, n_perm=10)
        else:
            calculate_pathway_gsea(matrix, {"P": ["target"]}, n_perm=10)


def test_unique_rankings_still_allow_repeated_pathway_members():
    stats = pd.Series([10., 3., 2., 1.], index=["target", "g2", "g3", "g4"])
    expected = fgsea_simple({"P": ["target"]}, stats, n_perm=30, rng=np.random.default_rng(1))
    actual = fgsea_simple({"P": ["target", "target"]}, stats, n_perm=30, rng=np.random.default_rng(1))
    assert_frame_equal(actual, expected)
    assert actual.iloc[0]["es"] == 1
