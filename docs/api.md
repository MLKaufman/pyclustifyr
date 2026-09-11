# API reference

[README](../README.md) · [Runnable examples](#runnable-examples) · [Compatibility notes](#compatibility-notes)

This reference describes the current Python API. All names below except those
under **Submodule utilities** are exported from `pyclustifyr`. Names beginning
with `_` are implementation details. R is neither required nor invoked at
runtime; it is used only by the optional development scripts that regenerate
reference fixtures.

## Contents

- [Data conventions](#data-conventions)
- [Classification](#classification)
- [AnnData integration](#anndata-integration)
- [Calls and metadata](#calls-and-metadata)
- [Cluster summaries](#cluster-summaries)
- [Marker tables and feature selection](#marker-tables-and-feature-selection)
- [Similarity and permutations](#similarity-and-permutations)
- [GSEA](#gsea)
- [Plotting](#plotting)
- [External references](#external-references)
- [Submodule utilities](#submodule-utilities)
- [Runnable examples](#runnable-examples)
- [Compatibility notes](#compatibility-notes)

## Data conventions

| Object | Shape and labels |
|---|---|
| Expression matrix | Numeric `pandas.DataFrame`; genes in the index, cells in columns. Wrap NumPy arrays with gene/cell labels first. |
| Expression reference | Numeric DataFrame; genes in the index, reference types in columns. |
| Similarity matrix | DataFrame; query clusters or cells in the index, reference types in columns. |
| Metadata | DataFrame indexed by cell IDs with a cluster column, or a Series/list of cluster assignments. |
| Wide marker lists | One reference type per column, gene names as values, missing entries for unequal list lengths. |
| Numeric marker reference | Genes in the index, reference types in columns; used by positive/negative marker scoring. |
| AnnData | Cells × genes in `.X`; `.obs_names` are cell IDs and `.var_names` are gene IDs. |

Use unique gene/cell identifiers, consistent identifier types, and numeric
expression values. Gene symbols are matched exactly; there is no case folding,
identifier conversion, library-size normalization, or automatic duplicate-gene
aggregation. GSEA explicitly rejects duplicate ranking identifiers.

**Alignment.** `clustify`, `average_clusters`, and `gene_pct_markerm` align indexed
metadata with expression columns by cell ID. Non-default indexed Series are
also aligned. Lists and metadata with the default `RangeIndex(0, n)` are
positional. Indexed IDs must be unique and match expression columns; mismatches
raise `ValueError`. An explicit `cell_col` in `average_clusters` takes precedence
over the metadata index. Lower-level functions accepting `cluster_ids` or
`clusters` expect positional vectors; `downsample_matrix` aligns indexed
metadata using the same rules as classification. Per-cell calls are joined back using the metadata index.

**Expression scale.** For mean aggregation, `if_log=True` means input values are
natural-log `log1p` values: `expm1` is applied before averaging. The mean is then
`log1p` transformed unless `output_log=False`. Thus `if_log=False` alone does
**not** request raw-scale output. Reference values and per-cell expression are
not automatically transformed by `clustify`. Use compatible scales for query
and reference, especially with Pearson, cosine, or KL-based scoring.

**Output and mutation.** Classification functions return new tables/lists;
AnnData classifiers return a copy by default. Metadata writeback replaces
existing prediction columns with the same names; `rename_prefix="ref1"` creates
`ref1_type` and `ref1_r` (or `ref1_rank` for consensus). Vector outputs follow
metadata row order and contain one call per cell, including repeated cluster
calls. Filtered-out clusters produce missing calls on writeback. The exception
to copy-style access is `object_data(..., slot="meta.data")`, which returns
`.obs` directly. Plotting functions draw on the returned or supplied axes.

**Missing values and randomness.** Undefined correlations remain missing;
`cor_to_call` excludes them from selection and marks all-missing rows unassigned.
Missing cluster assignments are grouped as `orig.NA` by `clustify`; standalone
`average_clusters` drops missing assignments. An `rng` argument expects a
`numpy.random.Generator`, not an integer seed. Pass `np.random.default_rng(42)`
for reproducibility; a generator's state advances when reused.


## Classification


### clustify

```text
clustify(
    input: pd.DataFrame,
    ref_mat: pd.DataFrame,
    metadata: pd.DataFrame | pd.Series | list | None = None,
    cluster_col: str | None = None,
    query_genes: list | None = None,
    per_cell: bool = False,
    n_perm: int = 0,
    compute_method: str = 'spearman',
    pseudobulk_method: str = 'mean',
    verbose: bool = True,
    rm0: bool = False,
    vec_out: bool = False,
    rename_prefix: str | None = None,
    threshold: float | str = 'auto',
    low_threshold_cell: int = 0,
    exclude_genes: list | None = None,
    if_log: bool = True,
    rng = None,
    return_pvalues: bool = False,
    **kwargs,
)
```

Compare expression against a reference. `input` and `ref_mat` use the shapes
above. Shared genes are selected in query order; `query_genes` restricts that
intersection and `exclude_genes` removes genes from it.

| Parameters | Behavior |
|---|---|
| `metadata`, `cluster_col` | Required for cluster analysis; provide `cluster_col` when metadata is a DataFrame. |
| `per_cell` | Skip cluster averaging and score individual expression columns. Metadata may be omitted. |
| `compute_method` | `"spearman"`, `"pearson"`, `"kendall"`, `"cosine"`, or `"kl_divergence"`. |
| `pseudobulk_method`, `if_log` | Aggregation method and input scale; see `average_clusters`. |
| `rm0` | Treat query zeros as missing and use pairwise complete observations. Supported only for Pearson, Spearman, Kendall; other methods raise `ValueError`. |
| `low_threshold_cell` | Exclude clusters with fewer than this number of cells; zero disables exclusion. |
| `n_perm`, `rng`, `return_pvalues` | With `n_perm > 0` and `return_pvalues=True`, return a dictionary containing `score` and `p_val` DataFrames. Requires `vec_out=False`. The legacy score-only `n_perm > 0` route emits `FutureWarning`; use `n_perm=0` for scores only. |
| `vec_out`, `threshold`, `rename_prefix` | Convert scores to calls and expand to metadata rows. `threshold="auto"` uses 75% of the global maximum score, rounded to two decimals. These options do not change the matrix when `vec_out=False`. |
| `verbose` | Print gene count and output dimensions. Small-cluster warnings are separate from verbosity. |
| `**kwargs` | Forwarded to similarity computation; `total_reads` and `max_kl` apply to KL scoring. This is not a general preprocessing-options dictionary. |

**Returns:** with `return_pvalues=True`, `{"score": scores, "p_val": pvalues}`
with matching labels and the same +1/tie handling as `permute_similarity`.
Otherwise, clusters × reference types, or cells × reference types with
`per_cell=True`; `vec_out=True` returns a list of per-cell calls. Raises
`ValueError` for unsupported methods, missing required metadata, unmatched cell
IDs, empty shared-gene sets, or empty query/reference columns. Missing
correlations are possible for constant vectors or insufficient observations.


### clustify_lists

```text
clustify_lists(
    input: pd.DataFrame,
    marker: pd.DataFrame,
    marker_inmatrix: bool = True,
    metadata: pd.DataFrame | pd.Series | list | None = None,
    cluster_col: str | None = None,
    if_log: bool = True,
    per_cell: bool = False,
    topn: int = 800,
    cut: float = 0,
    genome_n: int = 30000,
    metric: str = 'hyper',
    output_high: bool = True,
    vec_out: bool = False,
    rename_prefix: str | None = None,
    threshold: float = 0,
    low_threshold_cell: int = 0,
    verbose: bool = True,
    input_markers: bool = False,
    details_out: bool = False,
    **matrixize_kwargs,
)
```

Compare `input` expression against `marker` lists. `marker_inmatrix=True`
expects an already wide marker-list table; `False` runs `matrixize_markers`
(except for `posneg`). `**matrixize_kwargs` are arguments to that converter,
not arbitrary scoring/GSEA options. Conversion options raise `TypeError` when
conversion is disabled (`marker_inmatrix=True` or `metric="posneg"`). Unknown
metric names are rejected. `details_out` is supported only for hyper, jaccard,
and rank_distance/spearman.

`metadata`, `cluster_col`, `if_log`, `low_threshold_cell`, and `verbose` follow
the classification conventions. `topn` selects top-expression genes using
average ranks for ties; `cut` requires expression strictly greater than the
cutoff. `input_markers=True` bypasses averaging and binarization: supply a
numeric genes × query-columns matrix. This route is intended for comparison
metrics, not cluster-only consensus/pct/posneg workflows.

| `metric` | Score and output |
|---|---|
| `hyper` | Hypergeometric upper-tail p-values, Holm-adjusted across reference types within each query; uses `genome_n`. Default score is `-log10(p)`. |
| `jaccard` | Intersection/union of selected genes and each marker list. |
| `rank_distance` (alias `spearman`) | **Rank-order distance, not Spearman's correlation coefficient.** Sums absolute differences in shared-gene order; fewer than two shared genes gives NaN. Default score is global maximum distance minus distance. |
| `pct` | Mean fraction of cells detecting each marker gene, then summarized per cluster. Always produces cluster scores. |
| `posneg` | Correlate cells with a numeric positive/negative marker reference and average by cluster. Wide gene-name lists are converted using `pos_neg_marker`. |
| `gsea` | GSEA p-values on the binarized query, transformed to `-log10(p)` by default. This route currently uses 1,000 permutations and does not expose an `rng` argument. |
| `consensus` | Average ranks from hyper/jaccard/pct/posneg; returns a **call table**, with cluster column, `type`, and `rank`, rather than a similarity matrix. Tied types are joined by `__`. |

`per_cell=True` scores cells for comparison metrics. `pct`, `posneg`, and
`consensus` remain cluster-level. Consensus passes `topn`, `cut`, `if_log`,
`genome_n`, and `low_threshold_cell` to components where applicable.

`output_high=False` retains raw p-values/distances in matrix output. Call
selection still uses the higher-is-better scale, as does `threshold` (for
hyper/GSEA, `threshold=2` means `-log10(p) >= 2`). `vec_out=True` returns one
call per metadata row. `rename_prefix` controls output names during conversion.
Consensus does not apply a separate similarity threshold to its final rank table.

**Returns:** query × reference scores; a consensus call table; or per-cell
calls when `vec_out=True`. For hyper/jaccard/rank_distance (or spearman), `details_out=True` returns
`{"res": scores, "details": overlap_strings}` with comma-separated intersecting
genes. `details_out` and `vec_out` cannot both be true. Missing marker entries
are ignored. Invalid hypergeometric universe sizes raise `ValueError`.


## AnnData integration


### clustify_adata

```text
clustify_adata(
    adata: ad.AnnData,
    ref_mat: pd.DataFrame,
    cluster_col: str | None = None,
    query_genes: list | None = None,
    n_genes: int = 1000,
    use_var_genes: bool = True,
    layer: str | None = None,
    obj_out: bool = True,
    vec_out: bool = False,
    rename_prefix: str | None = None,
    threshold: float | str = 'auto',
    per_cell: bool = False,
    **kwargs,
)
```

AnnData wrapper for `clustify`. `adata.X` (or `adata.layers[layer]`) is
transposed into a genes × cells DataFrame, and `.obs` supplies metadata.
`ref_mat` remains a genes × reference-types DataFrame.

If `query_genes=None` and `use_var_genes=True`, use `.var["highly_variable"]`,
taking at most `n_genes` genes in var order; zero keeps all. If the field is
absent or selects no genes, use all shared genes. Explicit `query_genes`
overrides this selection. `cluster_col` is required for cluster analysis,
but can be omitted for `per_cell=True`. `threshold`, `rename_prefix`, and
`**kwargs` follow `clustify`.

**Returns:** an AnnData copy with `type`/`r` added to `.obs` when `obj_out=True`;
a similarity matrix when `obj_out=False`; a call list when `vec_out=True`
(takes precedence over `obj_out`). No correlation matrix is stored in `.uns`.
Sparse expression is densified before feature selection: account for the full
cells × genes memory cost. The original AnnData is not modified.
`return_pvalues=True` is forwarded to `clustify` and returns its dictionary;
it requires `n_perm > 0`, `obj_out=False`, and `vec_out=False`.


### clustify_lists_adata

```text
clustify_lists_adata(
    adata: ad.AnnData,
    marker: pd.DataFrame,
    cluster_col: str | None = None,
    layer: str | None = None,
    obj_out: bool = True,
    vec_out: bool = False,
    rename_prefix: str | None = None,
    threshold: float = 0,
    per_cell: bool = False,
    **kwargs,
)
```

AnnData wrapper for `clustify_lists`; `marker` is a wide marker list or
numeric positive/negative reference as described there. `layer` selects the
expression layer; `cluster_col` selects `.obs` labels. `**kwargs` forwards
marker scoring/conversion options, including `metric` and `output_high`.
`per_cell`, `threshold`, and `rename_prefix` follow marker classification.

**Returns:** an AnnData copy with `type`/`r` in `.obs`, or `type`/`rank` for
consensus. `obj_out=False` returns the underlying scores/call table/details
dictionary; `vec_out=True` returns calls and takes precedence. `pct`, `posneg`,
and consensus calls expand to every cell in the assigned cluster. Sparse
expression is densified. No automatic highly-variable gene selection occurs.


### object_data

```text
object_data(
    adata: ad.AnnData,
    slot: str = 'data',
    layer: str | None = None,
    n_genes: int = 1000,
)
```

Read an AnnData field:

| `slot` | Return |
|---|---|
| `data` or `counts` | New genes × cells DataFrame from `.X`, or `layers[layer]`. `counts` is an alias and does not automatically select a counts layer. |
| `meta.data` | The original `.obs` DataFrame, not a copy. |
| `var.genes` | List of highly-variable gene names; `n_genes` limits the list in var order, zero keeps all. Missing HVG annotation returns `[]`. |

A non-None `layer` selects expression regardless of `slot`. Unknown slots raise
`ValueError`; a nonexistent layer raises `KeyError`. Sparse data is densified.


### object_ref

```text
object_ref(
    adata: ad.AnnData,
    cluster_col: str | None = None,
    var_genes_only: bool = False,
    method: str = 'mean',
    if_log: bool = True,
)
```

Build a genes × clusters reference from `adata.X`, grouped by `.obs[cluster_col]`.
Supply `cluster_col` explicitly. `method` and `if_log` follow `average_clusters`;
mean output is log1p transformed. `var_genes_only=True` restricts to all annotated
highly-variable genes; unlike `clustify_adata`, absent HVG annotation gives an
empty gene selection rather than falling back. Returns a DataFrame.


### write_meta

```text
write_meta(
    adata: ad.AnnData,
    meta: pd.DataFrame,
)
```

Return a copy of `adata` with `.obs` replaced by `meta`. This replaces the
entire metadata table, not selected columns. Supply one row per cell with
unique IDs matching `adata.obs_names`. Rows are aligned to observation order;
missing, extra, or duplicate IDs raise `ValueError`. Expression and cell IDs
are preserved. Both the input AnnData and supplied metadata remain independent
of the returned metadata table.


## Calls and metadata


### cor_to_call

```text
cor_to_call(
    cor_mat: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    cluster_col: str | None = 'cluster',
    collapse_to_cluster: str | bool = False,
    threshold: float | str = 0,
    rename_prefix: str | None = None,
    carry_r: bool = False,
)
```

Select the maximum score in each row of `cor_mat`. The index is carried into
an output column named `cluster_col` (`None` uses `"cluster"`). Scores below
`threshold` become `"unassigned"`; equality passes. `"auto"` is 75% of the
maximum score across the entire matrix, rounded to two decimals.

**Returns:** a DataFrame with `[cluster_col, "type", "r"]`. NaN never competes
with valid scores; all-missing rows receive an unassigned call with NaN `r`.
Ties produce a `-CLASH!` suffix and retain the first tied type in reference
column order. `carry_r=True` changes the unassigned text to
`"r<THRESHOLD, unassigned"`. Returned row order need not match matrix row order;
join by the identifier column.

`rename_prefix` renames type and score columns. `collapse_to_cluster=False`
or `None` disables collapse; `True` groups by `cluster_col`, and a string
names the actual metadata grouping column. Metadata is matched by its unique
cell-ID index (or a separate explicit ID column named `cluster_col` if the
index does not match). Missing IDs/groups raise `ValueError`.
Collapsed output is `[grouping_column, "type", "sum", "n"]`; prefixing renames
`type`, `sum`, and `n`. Only nonmissing scores at or above the threshold vote.
Groups with no eligible cells remain unassigned with `n=0` and missing `sum`.
`carry_r` changes presentation only, never eligibility or the winning type.


### cor_to_call_rank

```text
cor_to_call_rank(
    cor_mat: pd.DataFrame,
    cluster_col: str = 'cluster',
    threshold: float | str = 0,
    rename_prefix: str | None = None,
    top_n: int | None = None,
)
```

Melt `cor_mat` to `[cluster_col, "type", "r", "rank"]`. Rank scores in descending
order per query using dense ranks (1 is best); ties share a rank. `threshold`
can be numeric or `"auto"` as above. Scores below it receive rank 100; NaN
scores have missing ranks. `top_n` keeps ranks at most that value, so ties can
return more than `top_n` types. `rename_prefix` renames type/r, not rank.
Returns a DataFrame suitable for `call_consensus`.


### cor_to_call_topn

```text
cor_to_call_topn(
    cor_mat: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    col: str = 'cluster',
    collapse_to_cluster: str | bool = False,
    threshold: float = 0,
    topn: int = 2,
)
```

Return top `topn` calls per row of `cor_mat`, with columns `[col, "type", "r"]`,
sorted by identifier and descending score. Ranking uses minimum ranks for ties,
so more than `topn` results can survive. Scores below numeric `threshold` are
labeled `"r<THRESHOLD, unassigned"`.

`collapse_to_cluster` follows the same convention as `cor_to_call`: `True`
uses `col` as the grouping column, a string names another group, and `False`
or `None` disables collapse. Match cell IDs using the metadata index, with a
separate explicit `col` ID column as a fallback. Eligible calls have nonmissing
scores at or above `threshold`. Count calls per type/group, select up to `topn`
types by count then score sum, and join back to IDs. Expanded output includes
`type2` (group), `sum`, `n`, and metadata columns.


### call_to_metadata

```text
call_to_metadata(
    res: pd.DataFrame,
    metadata: pd.DataFrame,
    cluster_col: str,
    per_cell: bool = False,
    rename_prefix: str | None = None,
)
```

Join `res` calls onto a **DataFrame** `metadata`. `res` must have its query ID
in the first column. In cluster mode, join on `cluster_col`; called cluster
IDs must exist in metadata. Missing metadata cluster values are matched to
`orig.NA`. In `per_cell=True`, join the first call column to the metadata index.
Per-cell call IDs must be unique.

`rename_prefix` renames type/r/rank outputs. Existing metadata columns with the
same output names are replaced. Returns a new DataFrame in metadata row order,
retaining original cluster values and including missing outputs for unmatched
cells/clusters. No input is modified.


### collapse_to_cluster

```text
collapse_to_cluster(
    res: pd.DataFrame,
    metadata: pd.DataFrame,
    cluster_col: str,
    threshold: float = 0,
)
```

Public alias of `classify.collapse_to_cluster_fn`. `res` has cell IDs in its
first column plus `type` and `r`; metadata is indexed by those cell IDs.
`cluster_col` identifies the grouping field. Choose the most frequent call per
cluster, breaking count ties by score sum. Returns
`[cluster_col, "type", "sum", "n"]`.

The numeric `threshold` excludes scores below it; missing scores never vote.
Type labels are not parsed to determine eligibility. Groups represented in
`res` with no eligible calls remain `unassigned`, with `n=0` and missing `sum`.
Grouping metadata must have unique matching cell IDs and nonmissing group labels.


### call_consensus

```text
call_consensus(
    list_of_res: list[pd.DataFrame],
)
```

Combine a list of `cor_to_call_rank` tables. The first two columns identify
query and type; the named `rank` column is averaged across tables. Select the
smallest average rank per query. Tied types are joined with `__`.
Returns `[query_column, type_column, "rank"]`. Supply matching query/type
coverage across component tables; absent rows are not assigned an automatic
penalty. Similarity `r` values do not enter the consensus calculation.


## Cluster summaries


### average_clusters

```text
average_clusters(
    mat: pd.DataFrame,
    metadata,
    cluster_col: str = 'cluster',
    if_log: bool = True,
    cell_col: str | None = None,
    low_threshold: int = 0,
    method: str = 'mean',
    output_log: bool = True,
    subclusterpower: float = 0,
    cut_n: int | None = None,
)
```

Aggregate genes × cells `mat` into genes × clusters. `metadata` and
`cluster_col` follow the alignment conventions; `cell_col` explicitly names a
metadata field identifying matrix columns.

| `method` | Aggregation |
|---|---|
| `mean` | Mean after optional `expm1` (`if_log`); optional `log1p` output (`output_log`). NaNs are skipped. |
| `median` | Median of supplied values. |
| `trimean` | 0.25 × Q1 + 0.5 × median + 0.25 × Q3. |
| `truncate` | 10% trimmed mean of supplied values. |
| `min`, `max` | Minimum/maximum of supplied values. |

`if_log` and `output_log` affect only mean aggregation. For raw means, set both
false. `low_threshold > 0` excludes smaller clusters and warns; zero retains
them but warns below ten cells. Missing cluster labels are dropped. DataFrame
categorical labels retain category order; other labels are sorted.
`subclusterpower > 0` applies `overcluster` first. `cut_n` zeroes genes ranked
below the top `cut_n` per output column, using average ranks for ties.
Returns a new DataFrame; unsupported methods or metadata mismatch raise
`ValueError`.


### percent_clusters

```text
percent_clusters(
    mat: pd.DataFrame,
    metadata,
    cluster_col: str = 'cluster',
    cut_num: float = 0.5,
)
```

Return a genes × clusters DataFrame of detection fractions on [0, 1].
Detection is `mat >= cut_num`, including equality. Missing expression is
excluded from each gene's denominator; an all-missing group can remain NaN.
`metadata`/`cluster_col` follow `average_clusters`. No log transform is applied.


### overcluster

```text
overcluster(
    mat: pd.DataFrame,
    cluster_id: dict[str, list],
    power: float = 0.15,
)
```

`cluster_id` maps cluster names to lists of matrix cell-column names. For each
group with more than one cell, run k-means on cells × genes with
`max(1, int(n_cells ** power))` centers and fixed seed 0. Return a dictionary
of subgroup names (`parent_label`) to cell IDs; singleton names are retained.
Requires numeric finite data suitable for k-means.


### get_best_match_matrix

```text
get_best_match_matrix(
    cor_mat: pd.DataFrame,
)
```

Return an integer DataFrame with the same labels/shape as `cor_mat`: 1 marks
each row maximum, including ties; other entries are 0. All-NaN rows contain
only zeros. This helper applies no assignment threshold.


### get_best_str

```text
get_best_str(
    name,
    best_mat: pd.DataFrame,
    cor_mat: pd.DataFrame,
    carry_cor: bool = True,
)
```

Format the row `name` of binary `best_mat`, using matching scores from
`cor_mat`. `carry_cor=True` produces strings such as `"B (0.82)"`; false
returns names only. Multiple matches are separated by `; `; no matches gives
`"?"`. Returns a string; matrix labels must match.


### assign_ident

```text
assign_ident(
    metadata: pd.DataFrame,
    clusters,
    idents,
    cluster_col: str = 'cluster',
    ident_col: str = 'type',
)
```

Return a metadata copy with `ident_col` overwritten for selected values of
`cluster_col`. `clusters` is a string or iterable; `idents` is a string or
iterable. A single ident is broadcast across requested clusters; otherwise
lengths must match or `ValueError` is raised. For a numeric single cluster,
pass a one-element list.


## Marker tables and feature selection


### matrixize_markers

```text
matrixize_markers(
    marker_df: pd.DataFrame,
    ranked: bool = False,
    n: int | None = None,
    step_weight: float = 1,
    background_weight: float = 0,
    unique: bool = False,
    remove_rp: bool = False,
)
```

Convert `marker_df` from long columns `gene`/`cluster`, or a wide gene-name
list. Long `feature`/`group` input is renamed and sorted by `padj` (required for
that form). Wide input preserves reference column order. Missing genes are
removed before counting; `remove_rp` removes genes matching the built-in
ribosomal-gene pattern. `unique=True` retains only genes appearing once in
the entire table.

Each nonempty group is truncated to the smallest group's length, additionally
capped by `n` if supplied. With `ranked=False`, return wide gene-name columns
and a 1-based position index. With `ranked=True`, return a numeric genes ×
types matrix, weighting positions by `step_weight * (length - zero_based_position)
+ background_weight`; absent gene/type pairs are zero. Empty input after
filtering returns an empty table with reference columns preserved. Numeric
ranked output is not a wide gene-name list for hyper/Jaccard comparison.


### binarize_expr

```text
binarize_expr(
    mat: pd.DataFrame,
    n: int = 1000,
    cut: float = 0,
)
```

Return a 0/1 DataFrame retaining genes whose expression is strictly greater
than `cut` and whose descending average rank is at most `n`. Ties may result
in fewer than `n` retained genes. If `n` is at least the gene count, only the
expression cutoff is applied. The input is not modified.


### compare_lists

```text
compare_lists(
    bin_mat: pd.DataFrame,
    marker_mat: pd.DataFrame,
    n: int = 30000,
    metric: str = 'hyper',
    output_high: bool = True,
    details_out: bool = False,
)
```

Compare columns of numeric `bin_mat` (genes × queries) against wide gene-name
columns of `marker_mat`. For hyper/Jaccard, selected query genes have value 1.
`n` is the hypergeometric universe size. `metric`, `output_high`, and
`details_out` have the comparison meanings documented under `clustify_lists`;
this helper does not support `pct`, `posneg`, or `consensus`.

**Returns:** query × marker-type scores, or `{"res": scores, "details": strings}`.
Hypergeometric p-values are Holm-adjusted within each query. Jaccard requires
nonempty unions. Rank-distance comparisons need at least two shared genes.
GSEA calls use 1,000 permutations.

Explicit metrics are never inferred or replaced. Hyper/Jaccard require every
query value to be 0 or 1; ranked or missing values raise `ValueError`. Request
`"rank_distance"` explicitly for ranked input (`"spearman"` remains a compatibility
alias for the same distance). Unknown metrics and invalid hypergeometric
universe sizes raise `ValueError`.


### get_vargenes

```text
get_vargenes(
    marker_mat: pd.DataFrame,
)
```

Extract gene names from a nonempty `matrixize_markers` result. A first index
label equal to `1` (as text) identifies unranked wide lists; otherwise genes
are taken from the index. Returns unique values in encounter order. This is a
format-specific helper; arbitrary DataFrame indexes are not reliably inferred.


### marker_select

```text
marker_select(
    row: pd.Series,
    cut: float = 1,
    compto: int = 1,
)
```

Inspect one numeric gene `row`, indexed by reference type. If the largest
value is at least `cut`, return `(best_type, runner_up / maximum)`, where
`compto=1` selects the second-highest value, 2 the third-highest, etc. Otherwise
return `None`. Supply at least `compto + 1` reference types and a nonzero
maximum. Smaller ratios indicate greater specificity.


### ref_marker_select

```text
ref_marker_select(
    mat: pd.DataFrame,
    cut: float = 0.5,
    arrange: bool = True,
    compto: int = 1,
)
```

Apply `marker_select` to every gene of genes × types `mat`, dropping missing
gene IDs and rows whose sum is zero. `cut` and `compto` control marker
selection; `arrange=True` sorts by cluster then ratio. Returns a DataFrame
with `gene`, `cluster`, `ratio`, retaining gene identifiers.


### pos_neg_marker

```text
pos_neg_marker(
    mat: pd.DataFrame,
)
```

Convert a wide marker-name table `mat` into a numeric genes × types DataFrame:
listed genes receive 1, other gene/type pairs 0. Missing entries are ignored
and repeated gene/type pairs collapse to one entry. Reference column order is
preserved.


### reverse_marker_matrix

```text
reverse_marker_matrix(
    mat: pd.DataFrame,
)
```

For each column of wide gene-name `mat`, list genes found elsewhere in the
matrix but not in that column. Return a wide table with original column names,
padding unequal list lengths with NaN. Drop missing entries when consuming the
returned lists.


### pos_neg_select

```text
pos_neg_select(
    input: pd.DataFrame,
    ref_mat: pd.DataFrame,
    metadata,
    cluster_col: str = 'cluster',
    cutoff_score: float | None = 0.5,
)
```

`input` is genes × cells; `ref_mat` is numeric genes × types. Compute per-cell
Spearman scores on shared reference genes, replace missing scores with zero,
and take their raw mean by `metadata`/`cluster_col`. Returns clusters × types.
A synthetic `clustifyr0` query gene with expression 0.01 is added; it affects
scoring only when also present in the reference.

For each type with maximum score greater than 0.1, `cutoff_score` zeroes
positive scores below that fraction of its maximum. `None` disables this
postprocessing. Cluster metadata follows the alignment conventions.


### gene_pct

```text
gene_pct(
    matrix: pd.DataFrame,
    genelist: list,
    clusters,
    returning: str = 'mean',
)
```

For genes in `genelist` that exist in genes × cells `matrix`, compute the
fraction of cells with expression > 0 within each positional `clusters` group.
`returning` is `"mean"`, `"min"`, or `"max"` across the genes' detection
fractions. Returns a Series indexed by cluster, preserving first-seen order.
No shared genes gives NaN; missing cluster labels become `orig.NA`.


### gene_pct_markerm

```text
gene_pct_markerm(
    matrix: pd.DataFrame,
    marker_m: pd.DataFrame,
    metadata,
    cluster_col: str | None = None,
    norm: str | float | None = None,
)
```

Apply `gene_pct` to each wide marker-list column of `marker_m`, using aligned
`metadata` and `cluster_col`. Return a clusters × marker-types DataFrame,
filling missing results with zero. `norm=None` leaves fractions unchanged;
`"divide"` divides each column by its maximum; `"diff"` subtracts that maximum;
a numeric value sets values above that fraction of the maximum to 1, others
to 0.


### ref_feature_select

```text
ref_feature_select(
    mat: pd.DataFrame,
    n: int = 3000,
    mode: str = 'var',
    rm_lowvar: bool = True,
)
```

Select up to `n` genes from genes × reference-types `mat`. With
`rm_lowvar=True`, first keep the floor of half the genes with highest sample
variance. `mode="var"` returns genes ordered by variance; `mode="cor"` orders
by each gene's maximum absolute Spearman correlation with another retained
gene. Returns a list of gene IDs. Supply enough reference types/genes for
variance and correlation to be defined; correlation mode constructs a dense
gene × gene matrix.


### feature_select_pca

```text
feature_select_pca(
    mat: pd.DataFrame | None = None,
    pcs: pd.DataFrame | None = None,
    n_pcs: int = 10,
    percentile: float = 0.99,
    if_log: bool = True,
)
```

Supply genes × samples `mat`, or precomputed genes × components `pcs`.
If `pcs` is supplied, it takes precedence. Otherwise apply `log1p` when
`if_log=False`, center each gene, and compute SVD loadings (no variance scaling).
For up to `n_pcs` available components, select genes at or above the
`percentile` quantile of absolute loadings. Returns a list that may repeat a
gene selected by multiple components; use `list(dict.fromkeys(result))` if a
unique list is needed. Inputs must be finite and percentile must be in [0, 1].


### append_genes

```text
append_genes(
    gene_vector: list,
    ref_matrix: pd.DataFrame,
)
```

Return `ref_matrix` reindexed into the order of `gene_vector`, adding all-zero
rows for missing genes and omitting genes not requested. Supply unique gene
identifiers to avoid duplicate-label ambiguity.


### check_raw_counts

```text
check_raw_counts(
    counts_matrix: pd.DataFrame,
    max_log_value: float = 50,
)
```

Heuristic label for numeric `counts_matrix`: returns `"raw counts"` if all
values are integers; otherwise `"normalized"` if the maximum exceeds
`max_log_value`; otherwise rejects negatives and returns `"log-normalized"`.
This is not a preprocessing validator: checks run in that order, so integer
or high-valued input can bypass negative-value detection. Choose `if_log`
from knowledge of the data's preprocessing, not this heuristic alone.


### make_comb_ref

```text
make_comb_ref(
    ref_mat: pd.DataFrame,
    if_log: bool = True,
    sep: str = '_and_',
)
```

Append a column for every unordered pair of types in `ref_mat`, named
`type1 + sep + type2`. With `if_log=True`, unlog with `expm1`, average each pair,
then log1p transform the combined original/pair matrix. With false, average
supplied values directly. Returns genes × (original types + pair types).


### downsample_matrix

```text
downsample_matrix(
    mat: pd.DataFrame,
    n: int | None = None,
    keep_cluster_proportions: bool = True,
    metadata = None,
    cluster_col: str = 'cluster',
    rng: np.random.Generator | None = None,
    *,
    frac: float | None = None,
    per_cluster: bool = False,
)
```

Sample columns without replacement using `rng`. Supply exactly one of:

- `n`: a nonnegative integer total cell count.
- `frac`: a finite fraction in [0, 1]; target count is `floor(frac * total_cells)`.

With `keep_cluster_proportions=False`, sample globally. With true, require
metadata and allocate the exact total across clusters proportionally using
largest remainders. Fractional ties go to clusters in first-seen expression
column order. Set `per_cluster=True` with `n` to select that many cells from
each cluster instead; this requires `keep_cluster_proportions=True`.

Indexed metadata is aligned by cell ID; default RangeIndex/plain vectors remain
positional. Invalid counts/fractions, missing assignments, nonunique cell IDs,
mismatched metadata, and counts exceeding available cells raise `ValueError`.
Returns the selected columns in sampling order. `n=1` now means one cell total;
use `frac=1` for all cells or `n=1, per_cluster=True` for one cell per cluster.


### calc_distance

```text
calc_distance(
    coord: pd.DataFrame,
    metadata,
    cluster_col: str = 'cluster',
    collapse_to_cluster: bool = False,
)
```

`coord` is cells × numeric coordinate dimensions, indexed by cell ID.
Compute pairwise Euclidean distances, then minimum distance from each cell to
each metadata cluster. `cluster_col` selects the grouping field. With
`collapse_to_cluster=False`, return cells × clusters; true returns a cluster ×
cluster matrix of minimum intercluster distances. Self-distances are included,
so distance to a cell's own cluster is zero. Memory use is quadratic in cells.


## Similarity and permutations


### clustifyr_methods

```text
clustifyr_methods = ("pearson", "spearman", "cosine", "kl_divergence", "kendall")
```

Tuple of supported expression similarity method names. This is distinct from
the marker-list `metric` choices.


### calc_similarity

```text
calc_similarity(
    query_mat: pd.DataFrame,
    ref_mat: pd.DataFrame,
    compute_method: str,
    rm0: bool = False,
    **kwargs,
)
```

Compare columns of `query_mat` and `ref_mat`; returns query columns × reference
columns. Every metric intersects gene IDs in query order and aligns both
matrices to that intersection. Gene IDs must be unique and nonmissing; no
shared genes raises `ValueError`. Extra genes are excluded consistently.

`compute_method` is one of `clustifyr_methods`. `rm0=True` removes query zeros
and missing values pairwise for Pearson/Spearman/Kendall; it is rejected for
cosine/KL. Without it, missing data can propagate to scores. `**kwargs` supplies
KL parameters (`if_log`, `total_reads`, `max_kl`). Unknown or inapplicable
keyword arguments raise `TypeError`, including options for correlation/cosine.
Constant vectors can return
NaN correlations; no calls or thresholds are applied.


### get_similarity

```text
get_similarity(
    expr_mat: pd.DataFrame,
    ref_mat: pd.DataFrame,
    cluster_ids,
    compute_method: str,
    pseudobulk_method: str = 'mean',
    per_cell: bool = False,
    rm0: bool = False,
    if_log: bool = True,
    low_threshold: int = 0,
    **kwargs,
)
```

Validate nonempty `expr_mat` and `ref_mat`, aggregate by positional
`cluster_ids`, then call `calc_similarity`. Genes must already be aligned.
`compute_method`, `pseudobulk_method`, `rm0`, `if_log`, `low_threshold`, and
`**kwargs` follow the corresponding similarity/aggregation routines.
`per_cell=True` skips aggregation; use expression cell IDs as `cluster_ids`.
Missing cluster IDs become `unknown` in this lower-level helper. Returns a
clusters/cells × reference-types DataFrame.


### cosine

```text
cosine(
    vec1: np.ndarray,
    vec2: np.ndarray,
)
```

Return the scalar dot-product cosine similarity of numeric arrays `vec1` and
`vec2`. Supply equal-length one-dimensional arrays with nonzero norms. This
helper performs no labeling, normalization beyond the cosine formula, or
missing-value filtering.


### kl_divergence

```text
kl_divergence(
    vec1: np.ndarray,
    vec2: np.ndarray,
    if_log: bool = False,
    total_reads: int = 1000,
    max_kl: float = 1,
)
```

Return a scalar similarity derived from shrinkage-estimated KL divergence.
`vec1`/`vec2` must have aligned features, nonnegative values after optional
`expm1` (`if_log=True`), and positive totals. Rescale each to `total_reads`,
round pseudo-counts, estimate divergence, and return `1 - 2 * KL / max_kl`.
`max_kl` must be positive. The score is **not clipped** and can be below -1.
For direct log-input handling, call this function or `calc_similarity` with
`if_log=True`; `clustify`'s `if_log` controls aggregation rather than this
per-vector transform.


## GSEA


### calc_gsea_stat

```text
calc_gsea_stat(
    ranks: np.ndarray,
    hit_positions: np.ndarray,
    gsea_param: float = 1,
)
```

Compute one weighted running-sum enrichment score. `ranks` is a one-dimensional
array of finite statistics **already sorted descending**. `hit_positions`
contains distinct zero-based integer positions of the pathway genes in that
array; the helper sorts these positions. `gsea_param` is the exponent applied
to absolute hit statistics. Supply at least one hit and fewer hits than genes;
selecting the whole universe raises `ValueError`. Returns a float: positive
means top enrichment, negative means bottom enrichment. No p-values or gene-ID
alignment are performed here.


### fgsea_simple

```text
fgsea_simple(
    pathways: dict[str, list],
    stats: pd.Series,
    min_size: int = 1,
    max_size: int | None = None,
    n_perm: int = 1000,
    gsea_param: float = 1,
    rng: np.random.Generator | None = None,
)
```

Preranked GSEA. `pathways` maps pathway names to gene lists; `stats` is a numeric
Series indexed by **unique** gene IDs and need not be sorted. Duplicate ranking
IDs raise `ValueError` before nonfinite values are filtered. Pathway members
are deduplicated; absent genes are excluded.

`min_size`/`max_size` filter by remaining overlap size. `max_size=None` allows
up to universe size minus one; full-universe and empty sets are skipped.
`n_perm` controls random sets sampled without replacement; use a positive
integer. Null sets have the same size as each pathway. `gsea_param` controls
weighting; `rng` controls reproducibility.

**Returns:** a DataFrame with `pathway`, `pval`, `es`, `nes`, `size`. The p-value
uses the null scores with the same sign as ES and a +1 correction. NES divides
ES by the mean magnitude of those same-sign scores. No multiple-testing-adjusted
p-value column is returned. Null samples are cached by set size **within one
call**. This implements simple permutation GSEA, not fgseaMultilevel.


### run_gsea

```text
run_gsea(
    expr_mat: pd.DataFrame,
    query_genes: list | dict[str, list],
    cluster_ids = None,
    n_perm: int = 1000,
    per_cell: bool = False,
    scale: bool = False,
    rng: np.random.Generator | None = None,
)
```

Run GSEA on genes × cells `expr_mat`. `query_genes` is a gene list (named
`query_genes` in output) or a pathway-name → gene-list dictionary. In cluster
mode, supply one positional `cluster_ids` entry per cell; expression is
aggregated with `average_clusters` defaults (log1p input/mean/log1p output).
`per_cell=True` scores columns directly and needs no cluster IDs.

`scale=True` standardizes each gene across input columns using sample standard
deviation **before** any cluster aggregation. For explicitly controlled
aggregation/scaling, aggregate first, then use `per_cell=True` or
`calculate_pathway_gsea`. `n_perm` and `rng` control null sampling.

**Returns:** a long DataFrame indexed by `cell` (cell or cluster ID), with
`pathway`, `pval`, and `nes`. Multiple pathways repeat index labels. Undefined
sets (no overlap or full finite universe) are retained with NaN p-value/NES.
Duplicate gene IDs are rejected. The input is not modified.


### calculate_pathway_gsea

```text
calculate_pathway_gsea(
    mat: pd.DataFrame,
    pathway_list: dict[str, list],
    n_perm: int = 1000,
    scale: bool = True,
    rng: np.random.Generator | None = None,
)
```

`mat` is an already aggregated genes × clusters/types matrix. `pathway_list`
maps pathway names to gene lists. Run per-column GSEA for each pathway and
return a **clusters/types × pathways NES matrix**. `scale=True` standardizes
each gene across columns; constant genes are excluded as nonfinite rankings.
`n_perm` and `rng` are forwarded. This helper returns NES only, not p-values.


### plot_pathway_gsea

```text
plot_pathway_gsea(
    mat: pd.DataFrame,
    pathway_list: dict[str, list],
    n_perm: int = 1000,
    scale: bool = True,
    topn: int = 5,
    returning: str = 'both',
    rng: np.random.Generator | None = None,
)
```

Compute the NES matrix using `calculate_pathway_gsea` with `n_perm`, `scale`,
and `rng`; keep the union of the top `topn` pathway calls per column of the
input `mat` for visualization. `pathway_list` maps names to gene lists.
`returning="both"` returns `(full_nes_matrix, ax)`; `"plot"` returns the heatmap
Axes; `"matrix"` returns the full NES matrix. The current implementation still
creates a figure in matrix mode. Missing scores are displayed as zero in the
heatmap, while the returned full matrix retains NaNs.


### gmt_to_list

```text
gmt_to_list(
    path: str,
    cutoff: int = 0,
    sep: str | None = None,
)
```

Read a local `.gmt` or `.gmt.gz` file at `path`. The default parser interprets
name, description, then gene fields separated by tabs. Description may be
`na`, a URL, empty, or plain text. Blank/comment lines are skipped; malformed
rows raise `ValueError` with the line number. Empty gene fields are removed.
`cutoff` retains lists with at least that many entries (zero disables filtering).

**Returns:** a dictionary of pathway name → gene list. `REACTOME_` is removed
from names for compatibility. An explicit `sep` enables the older custom-regex
split between pathway name and gene fields. Parsing itself does not deduplicate
genes; GSEA does that when scoring.


## Plotting


### plot_dims

```text
plot_dims(
    data: pd.DataFrame,
    x: str = 'UMAP_1',
    y: str = 'UMAP_2',
    feature: str | None = None,
    legend_name: str | None = None,
    cmap: str = pretty_palette2,
    d_cols: dict | list | None = None,
    pt_size: float = 5,
    alpha_col: str | None = None,
    group_col: str | None = None,
    scale_limits: tuple[float, float] | None = None,
    do_label: bool = False,
    do_legend: bool = True,
    do_repel: bool = True,
    ax: Axes | None = None,
)
```

Draw cells from DataFrame `data` using numeric coordinate columns `x` and `y`.
`feature=None` produces an uncolored scatter; a numeric feature uses `cmap` and
a colorbar, while other feature types use discrete categories. `legend_name`
overrides the feature label. `d_cols` supplies a category → color mapping or
color list for discrete features; otherwise the built-in `tab20` palette is used.
`pt_size` is scatter area. `alpha_col` min-max scales a metadata column to
opacity; a constant column gives opacity 1.

`scale_limits=(vmin, vmax)` controls numeric color range. `do_legend` enables
legend/colorbar. `do_label` adds labels at median coordinates; `group_col`
subdivides label groups. `do_repel` is accepted for compatibility but has no
effect. `ax` reuses an existing Axes; otherwise creates one. Returns that Axes.
Coordinates must already exist: this function does not compute UMAP/t-SNE.


### plot_cor

```text
plot_cor(
    cor_mat: pd.DataFrame,
    metadata: pd.DataFrame,
    data_to_plot: list | None = None,
    cluster_col: str | None = None,
    x: str = 'UMAP_1',
    y: str = 'UMAP_2',
    scale_legends: bool | tuple[float, float] = False,
    **kwargs,
)
```

Plot one embedding per reference type in `cor_mat`, using coordinates `x`/`y`
from `metadata`. `data_to_plot=None` selects all reference columns; otherwise
supply existing reference names. `cluster_col` joins cluster scores to metadata;
omit it for per-cell scores joined by metadata index. `scale_legends=True`
shares numeric limits across all plotted types; false scales separately; a
`(min, max)` tuple sets explicit limits. `**kwargs` forwards styling to
`plot_dims`. Returns a list of Axes in requested type order.


### plot_gene

```text
plot_gene(
    expr_mat: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: list,
    cell_col: str | None = None,
    **kwargs,
)
```

Plot expression for each requested `genes` entry present in genes × cells
`expr_mat`, joined to embedding `metadata`. `cell_col=None` joins by metadata
index; otherwise that column supplies cell IDs. `**kwargs` forwards coordinate
and style arguments to `plot_dims`. Missing genes are reported and skipped;
no requested genes present raises `ValueError`. Returns one Axes per retained
gene.


### plot_best_call

```text
plot_best_call(
    cor_mat: pd.DataFrame,
    metadata: pd.DataFrame,
    cluster_col: str = 'cluster',
    collapse_to_cluster: str | bool = False,
    threshold: float = 0,
    x: str = 'UMAP_1',
    y: str = 'UMAP_2',
    plot_r: bool = False,
    per_cell: bool = False,
    **kwargs,
)
```

Convert `cor_mat` to calls at `threshold`, join to metadata by `cluster_col`
(or index for `per_cell=True`), and color the `x`/`y` embedding by call.
`collapse_to_cluster=True` uses `cluster_col` for majority calls; a string names
another metadata grouping column. Cells retain their original coordinates.
`plot_r=True` returns `[type_ax, score_ax]`; otherwise returns `type_ax`.
The score plot retains per-cell scores when collapsing calls. `**kwargs`
forwards to `plot_dims`. Metadata containing `type` or `type2` is rejected;
rename/drop those columns before using this helper.


### plot_cor_heatmap

```text
plot_cor_heatmap(
    cor_mat: pd.DataFrame,
    cmap: str = not_pretty_palette,
    legend_title: str | None = None,
    ax: Axes | None = None,
)
```

Display numeric `cor_mat` with its index/columns as y/x labels. `cmap` selects
the colormap; `legend_title` labels the colorbar; `ax` optionally supplies an
existing Axes. Returns an Axes. Rows/columns are not clustered or reordered;
no normalization or significance filtering is applied.


## External references


### get_ucsc_reference

```text
get_ucsc_reference(
    cb_url: str,
    cluster_col: str,
    timeout: float = 60,
    **kwargs,
)
```

Download `meta.tsv` and `exprMatrix.tsv.gz` from a UCSC Cell Browser dataset.
`cb_url` must contain a `ds=` query value; literal `+` separates nested dataset
paths. `cluster_col` names the metadata annotation to aggregate.
`timeout` is passed to each HTTP request. `**kwargs` forwards to
`average_clusters`, notably `if_log`, `output_log`, and `method`; `cell_col`
overrides the first metadata column as the cell-ID field.

Cell IDs are read as strings, preserving leading zeros. Metadata is aligned
to expression columns, and missing/duplicate/mismatched IDs raise `ValueError`.
HTTP status failures are reported as `ValueError`; request/network failures may
propagate Requests exceptions. Returns a genes × annotated-types reference.
This helper needs network access and loads the full downloaded matrix into
memory. It does not write a Cell Browser export.


## Submodule utilities


These names are available through explicit imports, not `pyclustifyr.__all__`:

```python
from pyclustifyr.similarity import permute_similarity, vector_similarity
```


### permute_similarity

```text
permute_similarity(
    expr_mat: pd.DataFrame,
    ref_mat: pd.DataFrame,
    cluster_ids,
    n_perm: int,
    per_cell: bool = False,
    compute_method: str = 'spearman',
    pseudobulk_method: str = 'mean',
    rm0: bool = False,
    rng: np.random.Generator | None = None,
    if_log: bool = True,
    low_threshold: int = 0,
    **kwargs,
)
```

Return `{"score": observed_scores, "p_val": permutation_pvalues}` with matching
clusters/cells × reference-types labels. `expr_mat` and `ref_mat` must already
have unique gene IDs; shared genes are aligned by each similarity calculation. `cluster_ids` is positional cluster membership, or the
actual expression column names with `per_cell=True`.

`compute_method`, `pseudobulk_method`, `rm0`, `if_log`, `low_threshold`, and
`**kwargs` follow the similarity/aggregation routines. `n_perm` must be a
positive integer; `rng` controls shuffling. Cluster mode shuffles memberships
while preserving group sizes; per-cell mode reorders actual expression columns.
Observed and null scores use the same preprocessing and cluster exclusion.

P-values are `(1 + count(null >= observed)) / (n_perm + 1)`: ties count, identical
profiles yield 1, and finite runs cannot yield zero. Undefined observed scores
or any undefined null comparison yield NaN. These are unadjusted p-values;
no multiple-testing correction is applied.


### vector_similarity

```text
vector_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    compute_method: str,
    **kwargs,
)
```

Convert `vec1`/`vec2` to numeric arrays, require equal shapes, and dispatch
`compute_method="cosine"` or `"kl_divergence"`. `**kwargs` goes to
`kl_divergence`; cosine takes no additional parameters. Returns a float;
unsupported methods or mismatched shapes raise `ValueError`; unknown or
inapplicable keywords raise `TypeError`. Supply aligned
one-dimensional feature vectors.


## Runnable examples


Run these Python blocks **in order in one session**. They require no external
files, R installation, network access, or optional Scanpy dependency.

### 1. Shared data and expression classification

```python
import numpy as np
import pandas as pd
from pyclustifyr import clustify, cor_to_call, average_clusters

# Twenty cells, two clusters, six genes; raw (not log-transformed) counts.
genes = ["a1", "a2", "b1", "b2", "house1", "house2"]
cells = [f"cell{i}" for i in range(20)]
profile_a = [10, 8, 0, 0, 1, 2]
profile_b = [0, 0, 8, 10, 2, 1]
expr = pd.DataFrame(
    np.array([profile_a] * 10 + [profile_b] * 10).T,
    index=genes, columns=cells,
)
metadata = pd.DataFrame({"cluster": ["0"] * 10 + ["1"] * 10}, index=cells)
reference = pd.DataFrame({"A": profile_a, "B": profile_b}, index=genes)

scores = clustify(expr, reference, metadata=metadata, cluster_col="cluster",
                  if_log=False, verbose=False)
calls = cor_to_call(scores).set_index("cluster")
assert calls["type"].to_dict() == {"0": "A", "1": "B"}
assert scores.shape == (2, 2)
```

### 2. AnnData and marker lists

```python
import anndata as ad
from pyclustifyr import clustify_adata, clustify_lists

adata = ad.AnnData(expr.T, obs=metadata.copy())
annotated = clustify_adata(adata, reference, cluster_col="cluster",
                           if_log=False, verbose=False)
assert annotated.obs["type"].tolist() == ["A"] * 10 + ["B"] * 10
assert "type" not in adata.obs

markers = pd.DataFrame({"A": ["a1", "a2"], "B": ["b1", "b2"]})
marker_calls = clustify_lists(expr, markers, metadata=metadata,
                              cluster_col="cluster", if_log=False,
                              topn=2, vec_out=True, verbose=False)
assert marker_calls == ["A"] * 10 + ["B"] * 10
```

### 3. Permutations and standalone GSEA

```python
from pyclustifyr.similarity import permute_similarity
from pyclustifyr import fgsea_simple

permutations = permute_similarity(
    expr, reference, metadata["cluster"].tolist(), n_perm=50,
    if_log=False, rng=np.random.default_rng(42),
)
assert permutations["p_val"].shape == scores.shape
assert permutations["p_val"].to_numpy().min() >= 1 / 51

ranking = pd.Series([5., 4., 1., -1., -3., -4.], index=genes)
enrichment = fgsea_simple({"A_markers": ["a1", "a2"]}, ranking,
                          n_perm=100, rng=np.random.default_rng(42))
assert enrichment.iloc[0]["es"] > 0
assert enrichment.iloc[0]["size"] == 2
```

### 4. Plotting and saving a figure

```python
import matplotlib.pyplot as plt
from pyclustifyr import plot_best_call

embedding = metadata.assign(UMAP_1=np.arange(20), UMAP_2=np.repeat([0., 1.], 10))
ax = plot_best_call(scores, embedding, cluster_col="cluster")
assert sum(len(collection.get_offsets()) for collection in ax.collections) == 20
# To save: ax.figure.savefig("cell-types.png", dpi=150, bbox_inches="tight")
plt.close(ax.figure)
```


## Compatibility notes


- The R-derived tests compare selected routines against committed fixtures;
  Python tests do not launch R. Regenerating fixtures requires a separate
  R/clustifyR environment.
- Permutation similarity includes ties and a +1 correction, intentionally
  differing from the R strict-greater-than calculation.
- GSEA uses same-size random gene sets, caching null samples within one
  `fgsea_simple` call. It is not fgseaMultilevel and does not reproduce R's RNG
  or promise identical p-values/NES. Core enrichment-score fixtures check
  agreement on defined input cases.
- Duplicate GSEA ranking gene IDs are rejected; duplicate pathway members are
  deduplicated. Nonfinite rankings are excluded from both observed and null
  universes. Undefined pathways remain missing in wrapper output.
- Undefined correlation scores do not win calls. Rank-distance marker
  comparisons with fewer than two shared genes are missing.
- `ref_marker_select` retains gene names. `percent_clusters` returns actual
  fractions rather than log-transformed fractions.
- AnnData replaces R container interfaces; wrappers densify sparse expression.
  There is no automatic raw-count normalization or differential-expression
  marker discovery.
- `do_repel` has no effect in Matplotlib. Cell Browser export/building and
  GO-enrichment plotting are not implemented.

See the [README](../README.md) for installation and the [MIT license](../LICENSE).
