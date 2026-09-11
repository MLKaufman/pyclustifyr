# pyclustifyr

<p align="center">
  <img src="docs/assets/pyclustifyr-logo.png" alt="pyclustifyr logo: three connected cell clusters forming a C" width="320">
</p>

A Python port of the R package [clustifyr](https://github.com/rnabioco/clustifyr):
classify single-cell RNA-seq clusters (or individual cells) by correlating their
expression against reference expression data (bulk RNA-seq, sorted populations,
other scRNA-seq atlases) or against marker gene lists — no manual marker-gene
squinting required.

Works on plain `pandas`/`numpy` matrices or directly on `AnnData` objects.

## Installation

Install directly from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/MLKaufman/pyclustifyr
```

Or into an existing environment with `pip`:

```bash
pip install git+https://github.com/MLKaufman/pyclustifyr
```

> Replace the URL above with wherever this repo actually ends up hosted, if different.

To work on the package itself, clone it and let `uv` set up the environment:

```bash
git clone https://github.com/MLKaufman/pyclustifyr
cd pyclustifyr
uv sync
```

## Quick start

`clustify()` takes a genes × cells expression matrix, per-cell cluster
assignments, and a genes × cell-types reference matrix, and returns a
clusters × cell-types similarity matrix:

```python
import numpy as np
import pandas as pd
from pyclustifyr import clustify, cor_to_call

# expr_mat: genes x cells, metadata: cluster assignment per cell
expr_mat = pd.read_csv("expression.csv", index_col=0)       # genes x cells
metadata = pd.read_csv("metadata.csv", index_col=0)         # has a "cluster" column
ref_mat = pd.read_csv("reference.csv", index_col=0)         # genes x cell-types

similarity = clustify(
    expr_mat,
    ref_mat,
    metadata=metadata,
    cluster_col="cluster",
)
print(similarity.round(3))
#    T cell  B cell  NK cell  Monocyte
# 0   0.417  -0.172    0.090    -0.035
# 1   0.101   0.469   -0.072     0.021
# 2   0.011  -0.070    0.413     0.101

# Take the best-scoring call for each cluster
print(cor_to_call(similarity))
#   cluster     type         r
# 0       0   T cell  0.416540
# 1       1   B cell  0.469270
# 2       2  NK cell  0.413432
```

`compute_method` selects the similarity metric — `"spearman"` (default),
`"pearson"`, `"kendall"`, `"cosine"`, or `"kl_divergence"`. Pass
`per_cell=True` to classify individual cells instead of clusters, or
`vec_out=True` to get back a plain list of per-cluster/per-cell calls instead
of the similarity matrix.

### Runnable end-to-end example

The snippet below is self-contained (synthetic data, no files needed) — copy,
paste, and run it to see `clustify()` correctly recover three cell types from
noisy marker expression:

```python
import numpy as np
import pandas as pd
from pyclustifyr import clustify, cor_to_call

rng = np.random.default_rng(0)
genes = [f"gene_{i}" for i in range(60)]
cells = [f"cell_{i}" for i in range(180)]
cluster_of_cell = np.repeat(["0", "1", "2"], 60)

# Baseline noise everywhere, plus a boosted marker block per cluster
expr = pd.DataFrame(rng.poisson(1, size=(60, 180)), index=genes, columns=cells)
marker_blocks = {"0": genes[0:10], "1": genes[10:20], "2": genes[20:30]}
for cluster, block in marker_blocks.items():
    cols = [c for c, cc in zip(cells, cluster_of_cell) if cc == cluster]
    expr.loc[block, cols] += rng.poisson(15, size=(len(block), len(cols)))

metadata = pd.DataFrame({"cluster": cluster_of_cell}, index=cells)

# A reference where "T cell"/"B cell"/"NK cell" are driven by the same marker blocks
ref = pd.DataFrame(
    rng.poisson(1, size=(60, 4)),
    index=genes,
    columns=["T cell", "B cell", "NK cell", "Monocyte"],
)
ref.loc[genes[0:10], "T cell"] += 15
ref.loc[genes[10:20], "B cell"] += 15
ref.loc[genes[20:30], "NK cell"] += 15

similarity = clustify(expr, ref, metadata=metadata, cluster_col="cluster")
print(cor_to_call(similarity))
```

## Working with AnnData / scanpy

`clustify_adata()` pulls the expression matrix + `.obs` metadata out of an
`AnnData` object, runs `clustify()`, and writes the call back into `.obs`
(returning a new `AnnData`, matching clustifyr's Seurat/SingleCellExperiment
behavior):

```python
import scanpy as sc
from pyclustifyr import clustify_adata

adata = sc.read_h5ad("my_data.h5ad")
ref_mat = pd.read_csv("reference.csv", index_col=0)  # genes x cell-types

result = clustify_adata(
    adata,
    ref_mat,
    cluster_col="leiden",       # column in adata.obs with cluster assignments
    use_var_genes=True,          # restrict to adata.var["highly_variable"] if present
)
print(result.obs[["leiden", "type", "r"]].drop_duplicates())

# Get the raw similarity matrix instead of writing back into .obs:
similarity = clustify_adata(adata, ref_mat, cluster_col="leiden", obj_out=False)
```

`object_data(adata, slot=...)` (`"data"`, `"meta.data"`, `"var.genes"`),
`object_ref(adata, cluster_col=...)` (build a reference straight from an
already-annotated `AnnData`), and `write_meta(adata, meta)` are also available
for lower-level access.

## Classifying against marker gene lists

If you have marker gene lists instead of a full reference expression matrix,
`clustify_lists()` scores clusters by gene-set overlap (hypergeometric test by
default):

```python
from pyclustifyr import clustify_lists

markers = pd.DataFrame({
    "T cell": ["CD3D", "CD3E", "CD3G"],
    "B cell": ["MS4A1", "CD79A", "CD79B"],
    "NK cell": ["GNLY", "NKG7", "KLRD1"],
})

scores = clustify_lists(
    expr_mat,
    markers,
    metadata=metadata,
    cluster_col="cluster",
    metric="hyper",   # or "jaccard", "spearman", "pct", "posneg", "gsea", "consensus"
)
```

`clustify_lists_adata()` is the `AnnData`-native equivalent. With
`output_high=False`, score matrices retain raw p-values or rank distances;
call selection still chooses the strongest match. Call thresholds always use
the higher-is-better scale (for hyper/GSEA, `-log10(p)`). Percentage,
positive/negative, and consensus scores describe clusters; `vec_out=True`
expands those calls to cells in metadata order.

When writing predictions back to metadata, existing output columns with the
same names are replaced. Use `rename_prefix` to keep separate sets of predictions.

## Plotting

```python
from pyclustifyr import plot_dims, plot_best_call, plot_cor_heatmap

# Color cells on a UMAP/tSNE embedding by any metadata column
plot_dims(metadata, x="UMAP_1", y="UMAP_2", feature="cluster", do_label=True)

# Show the best-scoring call for each cluster on the embedding
plot_best_call(similarity, metadata, cluster_col="cluster")

# Heatmap of the clusters x cell-types similarity matrix
plot_cor_heatmap(similarity, legend_title="r")
```

Each function returns a matplotlib `Axes` (or a list of them), so you can
combine, save, or further style the plots as usual (`ax.figure.savefig(...)`).

## GSEA

Preranked gene set enrichment, either standalone or against reference-derived
pathway lists:

```python
from pyclustifyr import average_clusters, calculate_pathway_gsea, gmt_to_list

pathways = gmt_to_list("c2.cp.reactome.v6.2.symbols.gmt.gz")  # MSigDB-style .gmt(.gz)
cluster_avg = average_clusters(expr_mat, metadata, cluster_col="cluster")

nes = calculate_pathway_gsea(cluster_avg, pathways, n_perm=1000)
print(nes.round(2))  # clusters x pathways, normalized enrichment scores
```

> The core enrichment-score statistic matches R's `fgsea:::calcGseaStat` exactly.
> Permutation-based p-values/NES follow the same statistical design as fgsea but
> aren't pooled across same-size gene sets the way `fgseaMultilevel` is, so exact
> values won't bit-for-bit match R's RNG — they converge to the same numbers.

GSEA deduplicates pathway genes and excludes nonfinite ranking statistics
(including constant genes after scaling) from both scoring and the permutation
universe. Pathways with no remaining genes or covering the entire remaining
universe receive missing p-values/NES from `run_gsea()`.

## Building a reference from a UCSC Cell Browser dataset

```python
from pyclustifyr import get_ucsc_reference

ref_mat = get_ucsc_reference(
    "https://cells.ucsc.edu/?ds=cortex-dev",
    cluster_col="WGCNAcluster",
    if_log=False,
)
```

## Module overview

| Module | Contents |
|---|---|
| `similarity.py` | Similarity/correlation scoring: spearman, pearson, kendall, cosine, KL-divergence |
| `clusters.py` | `average_clusters` and other cluster-summary helpers |
| `classify.py` | `cor_to_call*`, `call_to_metadata`, `call_consensus` — turn similarity scores into calls |
| `clustify.py` | `clustify()` / `clustify_lists()` — the main entry points |
| `genelist.py` | Marker-list comparison: `compare_lists`, `matrixize_markers`, `binarize_expr` |
| `markers.py` | Marker/feature selection: `ref_marker_select`, `pos_neg_select`, `feature_select_pca`, ... |
| `anndata_io.py` | `AnnData` integration: `clustify_adata`, `object_data`, `object_ref`, `write_meta` |
| `plot.py` | matplotlib plotting: `plot_dims`, `plot_cor`, `plot_gene`, `plot_best_call`, `plot_cor_heatmap` |
| `gsea.py` | Preranked GSEA: `run_gsea`, `calculate_pathway_gsea`, `gmt_to_list`, `plot_pathway_gsea` |
| `cellbrowsers.py` | `get_ucsc_reference` — build a reference from a UCSC Cell Browser dataset |

## Tests

```bash
uv run pytest
```

Tests validate numerical parity against the R package directly — most results
match R to within floating-point precision (`~1e-9` to `~1e-16`). Where they
don't (and can't, in the case of GSEA's permutation RNG), it's called out in
the corresponding module and test file. `scripts/export_r_datasets.R` and
`scripts/export_r_expected.R` (re-)generate the fixtures under `tests/data/`
from the `clustifyr` R submodule.

## Relationship to clustifyr

This is a from-scratch Python port, not a wrapper — there's no R dependency at
runtime. It aims for numerical parity with the original wherever the algorithm
is well-defined, and documents where it deliberately deviates (e.g. it fixes a
gene-name-loss bug in `ref_marker_select` that exists in the R package). See
the original [clustifyr paper/repo](https://github.com/rnabioco/clustifyr) and
its authors (Fu, Riemondy, et al., RNA Bioscience Initiative) for the underlying
method.

**Not yet ported:** cellbrowser building/export *to* UCSC format (only
building a reference *from* a cellbrowser dataset is supported), and a couple
of GO-enrichment/rank-bias plotting helpers that depend on live internet
access to gene-ontology services.
