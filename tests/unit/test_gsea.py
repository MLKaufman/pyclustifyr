import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyclustifyr.clusters import average_clusters
from pyclustifyr.genelist import compare_lists
from pyclustifyr.gsea import calc_gsea_stat, calculate_pathway_gsea, gmt_to_list, plot_pathway_gsea, run_gsea
from tests.conftest import DATA_DIR


@pytest.fixture(scope="module")
def r_ranked_stats():
    # cross-checked against fgsea:::calcGseaStat with set.seed(1); rnorm(20), sorted descending
    vals = [
        1.59528080, 1.51178117, 1.12493092, 0.94383621, 0.82122120, 0.73832471, 0.59390132, 0.57578135,
        0.48742905, 0.38984324, 0.32950777, 0.18364332, -0.01619026, -0.04493361, -0.30538839, -0.62124058,
        -0.62645381, -0.82046838, -0.83562861, -2.21469989,
    ]
    genes = [f"g{i + 1}" for i in range(20)]
    return pd.Series(vals, index=genes)


def test_calc_gsea_stat_matches_r(r_ranked_stats):
    pathway = ["g2", "g5", "g9", "g13"]
    positions = np.array([r_ranked_stats.index.get_loc(g) for g in pathway])
    es = calc_gsea_stat(r_ranked_stats.to_numpy(), positions)
    assert es == pytest.approx(0.6349581, abs=1e-6)


def test_calc_gsea_stat_matches_r_negative_and_positive_cases():
    # cross-checked against fgsea:::calcGseaStat with set.seed(7); rnorm(50), sorted descending
    vals = [float(x) for x in (DATA_DIR / "gsea_stats_vals.txt").read_text().splitlines()]
    names = (DATA_DIR / "gsea_stats_names.txt").read_text().splitlines()
    stats = pd.Series(vals, index=names)

    pathway1 = [f"g{i}" for i in [40, 45, 48, 49, 50, 30, 35]]
    pathway2 = [f"g{i}" for i in [1, 2, 3, 4, 5, 10, 15]]
    hp1 = np.array([stats.index.get_loc(g) for g in pathway1])
    hp2 = np.array([stats.index.get_loc(g) for g in pathway2])

    assert calc_gsea_stat(stats.to_numpy(), hp1) == pytest.approx(-0.7767698795, abs=1e-6)
    assert calc_gsea_stat(stats.to_numpy(), hp2) == pytest.approx(0.8669182756, abs=1e-6)
    assert calc_gsea_stat(stats.to_numpy(), hp1, gsea_param=0.5) == pytest.approx(-0.7123102075, abs=1e-6)


def test_calc_gsea_stat_rejects_full_selection():
    stats = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        calc_gsea_stat(stats, np.array([0, 1, 2]))


def test_gmt_to_list_matches_r():
    path = DATA_DIR / "extdata" / "c2.cp.reactome.v6.2.symbols.gmt.gz"
    pathways = gmt_to_list(str(path))
    assert len(pathways) == 674
    first_key = next(iter(pathways))
    assert first_key == "GLYCOGEN_BREAKDOWN_GLYCOGENOLYSIS"
    assert pathways[first_key][:5] == ["AGL", "GYG1", "PGM1", "PHKA1", "PHKA2"]
    assert len(pathways[first_key]) == 18


def test_gmt_to_list_cutoff():
    path = DATA_DIR / "extdata" / "c2.cp.reactome.v6.2.symbols.gmt.gz"
    pathways = gmt_to_list(str(path), cutoff=10)
    assert len(pathways) == 671


def test_calculate_pathway_gsea_sign_matches_r(pbmc_matrix_small, pbmc_meta):
    # signs cross-checked against R's calculate_pathway_gsea() on the same data/pathways;
    # NES magnitude is stochastic (permutation-based) so only sign/rough order is checked here.
    gl = {"n": ["PPBP", "LYZ", "S100A9"], "a": ["IGLL5", "GNLY", "FTL"]}
    avg = average_clusters(pbmc_matrix_small, pbmc_meta, cluster_col="classified")
    rng = np.random.default_rng(42)
    res = calculate_pathway_gsea(avg, gl, n_perm=1000, rng=rng)

    expected_sign_n = {
        "Naive CD4 T": -1, "Memory CD4 T": -1, "CD14+ Mono": 1, "B": -1, "CD8 T": -1,
        "FCGR3A+ Mono": 1, "NK": -1, "DC": 1, "Platelet": 1,
    }
    for cluster, sign in expected_sign_n.items():
        assert np.sign(res.loc[cluster, "n"]) == sign, cluster


def test_run_gsea_requires_matching_cluster_ids(pbmc_matrix_small):
    with pytest.raises(ValueError):
        run_gsea(pbmc_matrix_small, ["PPBP"], cluster_ids=["a", "b"])


def test_compare_lists_gsea_metric_handles_zero_overlap(pbmc_matrix_small, pbmc_meta, cbmc_m):
    from pyclustifyr.genelist import binarize_expr

    avg = average_clusters(pbmc_matrix_small, pbmc_meta, cluster_col="classified")
    avgb = binarize_expr(avg)
    res = compare_lists(avgb, cbmc_m, metric="gsea")
    assert res.shape == (9, 13)
    assert res["Memory CD4 T"].isna().all()  # zero gene-universe overlap for this marker set


def test_plot_pathway_gsea(pbmc_matrix_small, pbmc_meta):
    gl = {"n": ["PPBP", "LYZ", "S100A9"], "a": ["IGLL5", "GNLY", "FTL"]}
    avg = average_clusters(pbmc_matrix_small, pbmc_meta, cluster_col="classified")
    res, ax = plot_pathway_gsea(avg, gl, n_perm=200, topn=1)
    assert res.shape == (9, 2)
    plt.close(ax.figure)
