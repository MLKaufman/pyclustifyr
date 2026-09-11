# API changes before 1.0

These changes resolve the [API stability review](api-stability-review.md).
They change some pre-1.0 behavior; the default classification return remains a
DataFrame and AnnData classifiers still return a copy by default.

| Previous behavior | New contract / migration |
|---|---|
| `write_meta` replaced row labels positionally | It aligns unique, matching cell IDs to AnnData observation order; missing/extra/duplicate IDs fail. Use a separate explicit operation to rename cells. |
| Ranked input could silently change an explicit overlap metric | `hyper`/`jaccard` require 0/1 input and reject other values. Use `rank_distance` for ranked input. `spearman` remains an alias for that distance, not a correlation coefficient. |
| `carry_r` changed which cells voted during collapse | Only nonmissing scores at or above threshold vote. Formatting does not affect eligibility. Groups with no eligible calls remain unassigned, with count zero and missing score sum. |
| Collapse grouping depended on the function | `False`/`None` disables collapse, `True` uses the default grouping column, and a string names the actual column. Cell IDs match the metadata index, with a separate explicit ID column as a fallback. |
| Unknown or inapplicable keywords could disappear | Similarity options and marker conversion options now fail explicitly when unused. Correct misspellings and only pass options supported by the selected algorithm. |
| `n_perm > 0` computed and discarded p-values | Set `return_pvalues=True` to receive `{"score": DataFrame, "p_val": DataFrame}`. Score-only permutation calls still work with `FutureWarning`; set `n_perm=0` when only scores are needed. |
| Some labeled numerical helpers were positional | Every similarity metric aligns the shared unique gene IDs. Downsampling aligns indexed metadata; ordinary vectors/default RangeIndex remain positional and must match cell count. |
| `downsample_matrix(n=1)` meant one cell per cluster and fractional `n` meant a fraction | Choose exactly one of integer `n` (total count) or `frac` (fraction). Use `n=1, per_cluster=True` for one cell per cluster and `frac=1` for every cell. |

## Explicit permutation results

```python
# expr/ref are labeled genes × cells/types DataFrames; meta is indexed by cell ID.
result = clustify(
    expr, ref, metadata=meta, cluster_col="cluster",
    n_perm=1000, return_pvalues=True, rng=np.random.default_rng(42),
)
scores, pvalues = result["score"], result["p_val"]
```

Use the same `if_log`, gene selection, aggregation, and filtering settings as
ordinary classification. `return_pvalues=True` requires positive `n_perm` and
`vec_out=False`. With `clustify_adata`, also set `obj_out=False`; p-values are
returned explicitly rather than silently written into `.obs` or `.uns`.
The existing `pyclustifyr.similarity.permute_similarity` entry point remains
available. P-values retain the +1 correction and inclusive tie handling.

## Sampling

```python
# Requires metadata when keep_cluster_proportions=True (the default).
subset = downsample_matrix(expr, n=100, metadata=meta, cluster_col="cluster")
subset = downsample_matrix(expr, frac=0.25, metadata=meta, cluster_col="cluster")
subset = downsample_matrix(expr, n=10, per_cluster=True, metadata=meta,
                           cluster_col="cluster")
```

Proportional sampling allocates the exact requested total using largest
remainders. Fractional ties use first-seen cluster order in the expression
matrix. A fraction requests `floor(frac * total_cells)` cells. Equal counts per
cluster require explicit `per_cluster=True`; oversize requests fail instead of
clipping. Pass a seeded NumPy generator for repeatability.

Unlike the old default, an omitted count/fraction is an error. No compatibility
warning can make an ambiguous silent sampling choice safe, so migrate each call
explicitly. Existing positional integer `n` arguments now mean a global count.

See the [API reference](api.md) for full signatures and return schemas.
