# API stability review before 1.0

Reviewed 2026-09-11. **Update: all eight findings below have been addressed.**
See [migration notes](api-migration.md) and the updated [API reference](api.md).
The original findings are retained below as the rationale for these changes. Scope: the 54 top-level exports and two documented similarity submodule
utilities, using the API reference, implementation inspection, and focused
executable reproductions. This does not establish correctness for every input
or parameter combination.

The reviewed API should not have been frozen unchanged. Three behaviors can silently change
cell identity, the requested statistic, or the final assignment. Five further
contracts should be settled before 1.0 to avoid later breaking changes. Several
are already documented limitations; they are identified here as design risks,
not newly discovered deviations from the documentation.

## Findings and proposed decisions

### 1. Preserve cell identity when writing metadata — P1

Location: [anndata_io.py](../src/pyclustifyr/anndata_io.py), `write_meta`, lines 52–56.

`write_meta(adata, adata.obs.iloc[::-1])` replaces the observation index without
reordering expression. In a two-cell example, cell `a` has expression 10 and cell
`b` has expression 20; the result labels expression 10 as cell `b`. Equal length
is insufficient to establish alignment. The API reference currently makes
correct order the caller's responsibility, but this is particularly hazardous
beside classifiers that align labeled metadata automatically.

**Proposed contract:** require unique, matching cell IDs; reindex metadata to
`adata.obs_names` before assignment. Reject missing, extra, or duplicate IDs.
Return an independent copy. Keep explicit cell renaming outside this helper.
Acceptance checks: reordered metadata preserves expression/identity association;
invalid IDs fail; editing the returned metadata does not mutate the input.

### 2. Honor the explicitly requested marker metric — P1

Location: [genelist.py](../src/pyclustifyr/genelist.py), `compare_lists`, lines 123–125.

If the first query column contains more than two distinct values, the function
replaces every metric except `gsea` with `spearman`. A query column `[3, 2, 1, 0]`
produces the same result with `metric="hyper"`, `metric="spearman"`, and even
`metric="typo"`. This also affects the `clustify_lists(input_markers=True)` route.
The returned table does not identify the substituted statistic.

**Proposed contract:** validate metric names first and never replace an explicit
choice. Require binary input for overlap metrics, with an actionable error for
ranked input. Expose the existing rank-distance algorithm as `rank_distance`;
retain `spearman` as a documented compatibility alias rather than silently
changing its algorithm. Automatic inference, if retained, needs an explicit
`metric="auto"` option. Test that changing the first column cannot change the
algorithm used for other columns.

### 3. Formatting must not affect cluster assignment — P1

Location: [classify.py](../src/pyclustifyr/classify.py), `cor_to_call` and
`collapse_to_cluster_fn`, especially line 90.

For one cluster with three cells whose best scores are 0.1, 0.1, and 0.9, using
threshold 0.5 and collapse enabled returns `unassigned` with `carry_r=False`,
but returns the accepted type with `carry_r=True`. The collapse routine excludes
only the formatted rejection string, so changing label presentation changes
which cells vote.

**Proposed contract:** decide eligibility from score/assignment status before
formatting labels. Reject below-threshold and undefined scores consistently;
format only the final output. Preserve clusters with no eligible cells as
unassigned. Test invariance under `carry_r`, including ties and all-rejected
clusters; do not rely on parsing user-facing type strings.

### 4. Unify collapse grouping parameters — P2

Location: [classify.py](../src/pyclustifyr/classify.py), `cor_to_call` lines 62–65
and `cor_to_call_topn` lines 141–145.

`cor_to_call(..., collapse_to_cluster="other")` ignores `other` and groups using
`cluster_col`. The top-N helper instead interprets the string as a metadata
column, despite both signatures accepting `str | bool`. In the reproduction,
`other` has three distinct groups but the result contains one `cluster` group.

**Proposed contract:** use one explicit grouping-column option, with `None`
disabling collapse, and one consistent rule for matching cell IDs. If existing
parameters are retained, `True` must mean the documented default group and a
string must name the actual group. Validate nonexistent columns before scoring.

### 5. Reject unused keyword arguments — P2

Locations: [clustify.py](../src/pyclustifyr/clustify.py), forwarding into
[similarity.py](../src/pyclustifyr/similarity.py), `calc_similarity` lines 122–166;
also the conditional `matrixize_kwargs` route in `clustify_lists`.

`clustify(..., compute_methd="pearson")` succeeds and uses the default Spearman
method. Correlation/cosine branches accept and discard arbitrary keywords.
Consequently, misspelled scientific options look accepted. Marker conversion
keywords are likewise unused when conversion is disabled.

**Proposed contract:** enumerate supported algorithm-specific options and reject
unknown or inapplicable arguments. Preserve forwarding in AnnData wrappers, but
validate at the receiving function. Test misspellings and valid KL options;
do not reject valid wrapper arguments merely because they use forwarding.

### 6. Give permutations an explicit result contract — P2

Location: [clustify.py](../src/pyclustifyr/clustify.py), lines 132–152.

Setting `n_perm=2` executes the permutations but still returns only the ordinary
score DataFrame, equal to the `n_perm=0` result. P-values are discarded at line
146. The usable permutation result exists only through the documented submodule
function. This behavior is documented, but it wastes computation and is an
awkward contract to promise indefinitely.

**Proposed contract:** keep the default classification return type unchanged.
Provide a top-level permutation entry point returning the existing explicit
`score`/`p_val` result and accepting the same labeled input preprocessing as
classification. Deprecate the score-only `n_perm` route with a clear migration
path. Alternatively, add an explicit result option and specify its interactions
with `vec_out` and AnnData writeback. Do not implicitly change return type just
because `n_perm` becomes positive.

### 7. Set one labeled-alignment rule for public numerical helpers — P2

Locations: [similarity.py](../src/pyclustifyr/similarity.py), `calc_similarity`
lines 150–170, and [markers.py](../src/pyclustifyr/markers.py), `downsample_matrix`
lines 265–272.

For the same labeled gene vectors `[1, 2, 4]`, reversing reference rows changes
Pearson similarity from 1.000 to -0.929. Correlation/cosine operate positionally,
while KL intersects gene labels. Separately, downsampling ignores metadata IDs
and `zip` silently truncates mismatched lengths: four expression cells and two
metadata rows return a plausible subset without reporting the missing labels.
These helpers differ from the main classifier's alignment behavior.

**Proposed contract:** exported DataFrame APIs align unique labels, or explicitly
reject a nonmatching index/order; they should never silently pair unrelated
labels. Keep positional behavior for array/vector interfaces. Downsampling
should use the shared metadata alignment/length validation helper. Test row
permutations, duplicate/missing labels, and mismatched metadata lengths.

### 8. Separate sampling fractions from counts — P2

Location: [markers.py](../src/pyclustifyr/markers.py), `downsample_matrix`
lines 249–278.

The default `n=1` selects one cell per cluster, while `n=0.99` selects about 99%
of each cluster. With integer `n`, `keep_cluster_proportions=True` selects the
same count from every cluster rather than preserving proportions. This is
documented, but the option names and discontinuity are likely to surprise users.

**Proposed contract:** separate `frac` and `n`, require exactly one, and define
whether `n` means a global target or a per-cluster count. If retaining a global
proportion-preserving mode, specify rounding/allocation and total output size.
Expose equal-count-per-cluster sampling as a distinct mode. Reject negative,
nonfinite, oversized, and nonintegral count requests explicitly.

## Contracts that can remain stable

- Labeled genes × cells DataFrames; AnnData uses cells × genes internally.
- A score matrix by default and one call per metadata cell for `vec_out=True`.
- AnnData wrappers return a copy by default; retain the documented output-option
  precedence and add tests for combinations rather than redesigning all returns.
- Natural-log `log1p` input semantics and explicit aggregation transforms.
- Undefined correlations remain missing; GSEA rejects duplicate ranking IDs.
- Seeded `numpy.random.Generator` arguments where currently exposed.
- MIT licensing and no R runtime dependency.

The varied marker return shapes (scores, overlap details, consensus call tables)
can remain if the existing contracts and incompatible options are validated.
A universal result class would be optional API design work, not a prerequisite
for correcting the findings above.

## Improvements that need not block 1.0

- Sparse processing and the unported plotting/export features can remain explicit
  limitations.
- Name consistency (`topn` versus `top_n`, `col` versus `cluster_col`) can use
  additive aliases; mass renaming is unnecessary.
- Mark plotting no-ops such as `do_repel` clearly, or warn when requested.
- Validate enum options such as `plot_pathway_gsea(returning=...)`; matrix-only
  output should eventually avoid creating a figure.
- Expose `rng`/`n_perm` for marker GSEA, which currently fixes permutations at
  1,000 and does not let callers supply a generator. The standalone GSEA API is
  an existing reproducible alternative.

## Resolution and release gate

All eight findings are addressed in the implementation and focused regression
tests in `tests/unit/test_api_contracts.py`. Permutations use the explicit
`return_pvalues` alternative described in finding 6; the legacy score-only
route warns instead of silently breaking existing callers. Sampling uses the
explicit count/fraction contract described in finding 8.

Before tagging 1.0, obtain a successful GitHub CI matrix run on Linux, macOS,
and Windows, then finalize release notes/version and validate release artifacts.
The 54 exports and two documented submodule utilities remain the supported
surface. Underscore-prefixed helpers remain internal.
