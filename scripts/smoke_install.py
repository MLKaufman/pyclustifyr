"""Exercise a built installation; run with its Python using -I, outside the repo."""

from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

import pyclustifyr as pc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    dist = distribution("pyclustifyr")
    installed_module = Path(dist.locate_file("pyclustifyr/__init__.py")).resolve()
    assert Path(pc.__file__).resolve() == installed_module, "Imported source checkout"
    assert installed_module.is_file(), "Wheel is missing the package"
    assert dist.metadata["License-Expression"] == "MIT"
    for name in pc.__all__:
        assert hasattr(pc, name), f"Missing public export: {name}"

    genes = ["a1", "a2", "b1", "b2", "house1", "house2"]
    cells = [f"cell{i}" for i in range(20)]
    a, b = [10, 8, 0, 0, 1, 2], [0, 0, 8, 10, 2, 1]
    expr = pd.DataFrame(np.array([a] * 10 + [b] * 10).T, index=genes, columns=cells)
    meta = pd.DataFrame({"cluster": ["0"] * 10 + ["1"] * 10}, index=cells)
    ref = pd.DataFrame({"A": a, "B": b}, index=genes)
    scores = pc.clustify(
        expr, ref, metadata=meta, cluster_col="cluster", if_log=False, verbose=False
    )
    calls = pc.cor_to_call(scores).set_index("cluster")
    assert calls["type"].to_dict() == {"0": "A", "1": "B"}

    permutations = pc.clustify(
        expr,
        ref,
        metadata=meta.iloc[::-1],
        cluster_col="cluster",
        if_log=False,
        verbose=False,
        n_perm=5,
        return_pvalues=True,
        rng=np.random.default_rng(42),
    )
    pd.testing.assert_frame_equal(permutations["score"], scores)
    assert permutations["p_val"].to_numpy().min() >= 1 / 6
    assert pc.downsample_matrix(expr, frac=0.5, metadata=meta).shape == (6, 10)

    markers = pd.DataFrame({"A": ["a1", "a2"], "B": ["b1", "b2"]})
    assert (
        pc.clustify_lists(
            expr,
            markers,
            metadata=meta,
            cluster_col="cluster",
            if_log=False,
            topn=2,
            vec_out=True,
            verbose=False,
        )
        == ["A"] * 10 + ["B"] * 10
    )

    adata = ad.AnnData(expr.T, obs=meta.copy())
    result = pc.clustify_adata(
        adata, ref, cluster_col="cluster", if_log=False, verbose=False
    )
    assert result.obs["type"].tolist() == ["A"] * 10 + ["B"] * 10
    assert "type" not in adata.obs
    aligned = pc.write_meta(adata, adata.obs.iloc[::-1])
    pd.testing.assert_frame_equal(aligned.obs, adata.obs)
    np.testing.assert_array_equal(aligned.X, adata.X)

    enrichment = pc.fgsea_simple(
        {"A_markers": ["a1", "a2"]},
        pd.Series([5.0, 4.0, 1.0, -1.0, -3.0, -4.0], index=genes),
        n_perm=100,
        rng=np.random.default_rng(42),
    )
    assert enrichment.iloc[0]["es"] > 0
    assert np.isfinite(enrichment.iloc[0]["nes"])
    assert 0 < enrichment.iloc[0]["pval"] <= 1

    with TemporaryDirectory() as directory:
        root = Path(directory)
        result.write_h5ad(root / "annotated.h5ad")
        restored = ad.read_h5ad(root / "annotated.h5ad")
        assert restored.obs["type"].tolist() == result.obs["type"].tolist()
        ax = pc.plot_cor_heatmap(scores)
        ax.figure.savefig(root / "scores.png")
        plt.close(ax.figure)
        assert (root / "scores.png").stat().st_size > 0
    print(
        f"Installed pyclustifyr {dist.version}: classification, markers, AnnData I/O, GSEA, plotting passed"
    )


if __name__ == "__main__":
    main()
