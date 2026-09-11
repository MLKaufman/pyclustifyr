# Changelog

User-facing changes are recorded here. See the [API reference](docs/api.md) for
current behavior and the [migration guide](docs/api-migration.md) for changes to
existing code.

## 1.0.0 — 2026-09-11

First stable release of the Python implementation, including the fixes and
API changes developed during the pre-1.0 review.

### Added

- Reference-based annotation of clusters or individual cells using Spearman,
  Pearson, Kendall, cosine, or KL-derived similarity.
- Marker-list classification using hypergeometric enrichment, Jaccard overlap,
  rank distance, detection percentages, positive/negative markers, GSEA, and
  consensus ranks.
- AnnData integration, cluster summaries, marker and feature selection,
  preranked GSEA, plotting, GMT import, and UCSC Cell Browser reference loading.
- Explicit permutation results through `clustify(return_pvalues=True)`, returning
  `score` and `p_val` DataFrames. AnnData callers can obtain the same result with
  `obj_out=False`. Both require positive `n_perm` and `vec_out=False`.
- A complete API reference, runnable examples, API migration guide, and README
  logo. Package metadata now includes a description, project links, and keywords.
- A GitHub Actions matrix for Python 3.11–3.14 on Linux, macOS, and Windows.
  Each job runs locked tests, builds a source distribution and wheel, and
  smoke-tests the installed wheel in a separate environment outside the checkout.
- Regression coverage for statistical edge cases, cell/gene alignment, output
  modes, and the public API contracts.

### Changed — migration required

- `write_meta` aligns metadata by unique, matching cell IDs instead of replacing
  row labels positionally. Missing, extra, or duplicate IDs raise an error.
- All expression similarity metrics align shared gene IDs in query order.
  Duplicate or missing gene IDs and an empty intersection are rejected.
- Indexed metadata is aligned to expression columns in classification and
  downsampling. Plain vectors and default RangeIndex metadata remain positional
  and must match the cell count.
- `compare_lists` honors the requested metric. Hypergeometric/Jaccard input must
  contain only 0 and 1; ranked input no longer silently changes the algorithm.
  Use `rank_distance` for ranked input. `spearman` remains a compatibility alias
  for this distance, not a Spearman correlation coefficient.
- Collapse options consistently interpret `True` as the default grouping column,
  a string as a specific grouping column, and `False`/`None` as disabled.
  Missing IDs or grouping fields fail explicitly.
- Unknown or inapplicable similarity keywords and unused marker-conversion
  options now raise errors instead of being silently ignored.
- `downsample_matrix` requires exactly one of integer `n` (total count) or `frac`
  (fraction of all cells). Proportional sampling allocates the exact total using
  largest remainders. Use `n=1, per_cluster=True` for one cell per cluster,
  `frac=1` for all cells, and `n=1` for one cell total. Invalid or oversized
  requests raise errors.

### Fixed

- Metadata replacement and reordered cell metadata no longer silently attach
  observations or predictions to the wrong cells. Prediction writeback replaces
  existing prediction columns and preserves metadata order.
- Undefined correlations cannot win cell-type assignments. All-missing rows
  remain unassigned, and rank-distance comparisons with insufficient shared
  genes remain missing.
- Cluster-collapse eligibility depends on numeric score thresholds, not formatted
  type strings. `carry_r` no longer changes the winner. Clusters with no eligible
  calls remain unassigned with count zero and a missing score sum; unused
  categorical groups do not create artificial winners.
- Consensus classification combines ranks rather than raw scores and retains
  tied types. Marker call thresholds use the higher-is-better scale even when
  raw p-values or distances are returned.
- Permutation tests count ties in the upper tail and apply `(k + 1)/(n_perm + 1)`.
  Identical profiles yield p=1; undefined comparisons yield missing p-values.
  Observed and null scores use consistent preprocessing and cluster filtering.
- GSEA rejects duplicate ranking gene IDs, including duplicates whose statistics
  would otherwise be filtered out. Repeated pathway members are deduplicated;
  nonfinite rankings are excluded from scoring and the null universe. Numeric
  input types are handled without unsigned-ranking overflow, and undefined
  pathways retain missing results in wrapper output.
- Pairwise zero/missing-value handling respects the selected correlation method.
  Detection percentages remain fractions on [0, 1], with missing expression
  excluded from denominators. Feature filtering preserves exclusions and gene
  names.
- UCSC reference loading aligns metadata by cell ID, preserves string IDs such
  as leading-zero identifiers, and rejects ambiguous IDs. GMT parsing supports
  standard descriptions, compressed files, and actionable malformed-row errors.

### Deprecated

- `clustify(n_perm > 0)` without `return_pvalues=True` still returns scores but
  emits `FutureWarning` because it discards the computed p-values. Request the
  explicit result or use `n_perm=0` when only scores are needed. The existing
  `pyclustifyr.similarity.permute_similarity` function remains available.

### Compatibility and limitations

- Python 3.11 or newer is required. Runtime computation is implemented in Python
  with the declared dependencies; R is neither required nor invoked. Only the
  optional development scripts that regenerate R fixtures require R/clustifyR.
- Licensing is MIT, matching upstream clustifyR, with its copyright notice
  retained in [LICENSE](LICENSE).
- Simple permutation GSEA is not `fgseaMultilevel` and does not promise identical
  R p-values or NES. Similarity permutation tie handling and the +1 correction
  intentionally differ from upstream's strict-greater-than calculation.
- AnnData wrappers densify sparse expression before feature selection. Large
  datasets must fit in memory; raw-count normalization and differential-expression
  marker discovery are not automatic.
- Cell Browser export/building and GO-enrichment plotting are not implemented.
  Matplotlib's `do_repel` option currently has no effect.

### Validation recorded before release

At the API-stability checkpoint, all 262 tests passed on each of Python
3.11–3.14 on macOS. Isolated package builds, the installed-wheel smoke check,
and API documentation examples also passed. The pre-release GitHub CI matrix
also passed all 12 Python/operating-system combinations. The versioned release
commit is checked again before tagging.
