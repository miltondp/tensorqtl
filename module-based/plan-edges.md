# Edge-level (locus A → expression B) QTL mapping for gene modules

## Context

`module-based/module-based-approach.md` describes building a **directed genetic-association
network** from a gene co-expression module: for every ordered gene pair (A, B), test all variants
in A's ±1 Mb window against the expression of B, and draw an edge A→B if there is at least one
significant association. Module genes are often on different chromosomes, so these are mostly
*trans* associations.

Step 2 of that document — **within-edge multiple testing** — needs one calibrated, LD-aware
p-value per edge: the min-p statistic over A's window, corrected by permutation +
Beta-approximation (FastQTL / `map_cis` style). `trans.map_trans` cannot provide this (nominal
p-values only). This plan implements step 2 plus the BH step 3, as a standalone module under
`module-based/`, reusing tensorQTL's existing machinery. **No files under `tensorqtl/` are
modified** — this fork must stay mergeable with upstream (CLAUDE.md).

### Facts established by reading the code

- **`cis.map_cis` already does this, unchanged, via pseudo-phenotypes.**
  `genotypeio.InputGeneratorCis` selects variants purely from `phenotype_pos_df`
  (`get_cis_ranges`, `tensorqtl/genotypeio.py:380`); it never inspects which gene a phenotype row
  belongs to. Feed it rows whose *values* are B's expression and whose *position* is A's TSS and
  it tests A's window against B. Only A's chromosome needs genotypes, so B elsewhere is fine.
- **The permutation machinery is position-agnostic.** `cis.calculate_cis_permutations`
  (`tensorqtl/cis.py:44-67`) takes only tensors and returns `r2_perm` = max r² over variants per
  permutation — the min-p null. Nothing about cis.
- **`pval_beta` is mandatory, not optional.** `pval_perm` floors at `1/(nperm+1)`
  (`cis.py:567`) = 1e-4 at nperm=1e4, while BH at 5% over E edges needs p ≈ 0.05/E — 5e-6 at
  M=100, 5e-8 at M=1000. With permutation p-values alone a module of ≥ ~30 genes yields zero
  discoveries.
- **The permutation null depends on B's expression only through its multiset of values.**
  Because the null statistic is `max_v r²(g_v, π(y))`, two phenotypes with the same sorted value
  vector have *identical* null distributions. This is what makes the O(M) shortcut possible —
  and, done via multiset bucketing, it is **exact**, not an approximation.
  (Upstream's `trans.map_permutations`, `tensorqtl/trans.py:336-342`, uses the special case
  "everything is `norm.ppf(i/(n+1))`", i.e. assumes a single bucket.)

### One correction to the obvious shortcut

Assuming every gene's marginal is exactly `norm.ppf(arange(1,n+1)/(n+1))` is **not safe on real
data**. A scan of the bundled expression BED (19,836 genes × 445 samples) found ~71% of genes
matching that vector to ~1e-16 (GTEx's inverse-normal transform produces it exactly) but ~29%
with **tied values** from low-count genes, whose sorted vectors deviate substantially. On chr18,
101 of 301 genes are tied. For those B's, a `norm.ppf`-based null is systematically miscalibrated.

**Bucketing by the sorted-value multiset fixes this at no cost**: one null per
(gene A, marginal bucket), permuting *that bucket's own* sorted values. A clean
inverse-normal-transformed module collapses to one bucket → full O(M) speedup **and** an exact
null for every B. A module with ties degrades gracefully toward per-pair. This subsumes the
`norm.ppf` shortcut and needs no normality assumption.

(These marginal statistics were produced by a subagent and should be re-derived as verification
step 0 — but the bucketing design is the right choice regardless, since it is exact either way.)

## Decisions made

- Build **both** paths: a reference (thin wrapper over `cis.map_cis`) as ground truth, and a fast
  one for production.
- **Exclude** pairs where A and B are on the same chromosome within a configurable distance
  (default 5 Mb) to avoid cis leakage; report the exclusions explicitly. Optionally also support
  per-pair variant masking (`cis_window_b`) — the design below handles it without giving up the
  shared null.
- Code in `module-based/`, importing the installed `tensorqtl` package.

---

## Step 0: environment

No conda env on this machine has `torch`, `pgenlib`, or `tensorqtl`
(`~/software/miniforge3/envs/*`: only `clamp-analyses` has numpy/pandas/scipy), and
`import tensorqtl` currently fails on `pandas`. There is also **no GPU** (`nvidia-smi` absent);
12 cores, 39 GB RAM. Before anything runs:

```bash
mamba env create -f install/tensorqtl_env.yml && conda activate tensorqtl
pip install -e .    # the yml pip-installs upstream tensorqtl; this overrides it with the fork
```

## Files to create

| File | Purpose |
|---|---|
| `module-based/module_edges.py` | Library: `build_pairs`, `build_pair_frames`, `map_pairs_reference`, `map_pairs`, `bh_edges`. |
| `module-based/verify_module_edges.py` | Verification suite against `example/data/` (chr18 GEUVADIS). |

Import convention — reach core primitives **through `cis`**, since `cis.py:12` does
`from core import *`, guaranteeing the same function objects `map_cis` uses and avoiding the
`core` vs `tensorqtl.core` duplicate-module hazard created by the flat imports:

```python
from tensorqtl import cis, trans, genotypeio, pgen, eigenmt
# cis.Residualizer core.py:62   cis.impute_mean core.py:124   cis.calculate_maf core.py:78
# cis.get_allele_stats core.py:84   cis.center_normalize core.py:135
# cis.calculate_corr core.py:141   cis.pval_from_corr core.py:339
# cis.fit_beta_parameters core.py:358   cis.output_dtype_dict core.py:23
```

---

## 1. Pair construction

```python
def build_pairs(module_genes, phenotype_pos_df, min_pair_distance=5_000_000, include_self=False):
    """All ordered (A,B) pairs, with same-chromosome/nearby pairs flagged for exclusion."""
```

Returns `['gene_a','gene_b','same_chr','gene_distance','excluded']` with
`excluded = same_chr & (gene_distance < min_pair_distance)`. Distance uses `pos` when present,
else `start`/`end` (both shapes come out of `core.read_phenotype_bed`, `core.py:426-427`).
`include_self=True` keeps (A,A) pairs for the verification suite. Callers pass
`pair_df[~pair_df['excluded']]` onward; both mappers assert nothing excluded slipped through.

Mappers also emit `same_chrom` and (when masking is used) `leakage_removed = nvar_full - num_var`
so §4B of the design doc ("label them separately") is satisfiable without rerunning, and a
`lead_variant_in_cis_of_b` flag via `trans._in_cis` (`tensorqtl/trans.py:16`) as a safety net.

---

## 2. Reference implementation (ground truth)

```python
def build_pair_frames(expression_df, phenotype_pos_df, pairs, sep='|', dtype=np.float32):
    """pair_expr_df.loc[f"{A}|{B}"] == expression of B;  pair_pos_df.loc[...] == position of A."""
```

Assertions that matter:
- `expression_df.index.equals(phenotype_pos_df.index)` — **see trap T1**, this is the
  highest-consequence failure mode in the whole design.
- gene IDs must not contain `sep`; pair index must be unique (`genotypeio.py:433`).
- sort pairs by (A's chrom, A's pos) so A's blocks are visited in genomic order.

```python
def map_pairs_reference(genotype_df, variant_df, expression_df, phenotype_pos_df,
                        covariates_df=None, pairs=None, **kwargs):
    pair_expr_df, pair_pos_df, pair_df = build_pair_frames(expression_df, phenotype_pos_df, pairs)
    kwargs.setdefault('warn_monomorphic', False)   # else one warning per pair
    res_df = cis.map_cis(genotype_df, variant_df, pair_expr_df, pair_pos_df,
                         covariates_df=covariates_df, **kwargs)     # cis.py:627, unmodified
    # reconcile against the requested pairs -- see trap T6 -- then split pair_id -> gene_a/gene_b
```

Everything comes free from `prepare_cis_output` (`cis.py:563-600`): `num_var` = post-filter
variants in **A's** window, `start_distance`/`end_distance` = lead variant − **A's** TSS
(`cis.py:722-723`, always within ±window so the `np.int32` dtype at `core.py:30-31` is safe),
`pval_perm`, `pval_beta`.

This path is O(M²) in both permutations and `fit_beta_parameters` calls — a **validation tool**,
not production.

---

## 3. Fast implementation

Two structural improvements over the naive approach:

**(a) Drive `InputGeneratorCis` with the real M×N expression matrix, not pseudo-phenotypes.**
Pass `expr_df` (module genes × samples) and their own `pos_df`. Each iteration then yields
`(expr_of_A, A's genotype block, genotype_range, gene_a)` — exactly one iteration per gene A,
with no M(M−1)×N matrix ever materialised. All B's live on the device as one M×N tensor.

**(b) Key the permutation null on `(exclusion interval, marginal bucket)`.** Cross-chromosome
pairs — the overwhelming majority — all share one null per A.

```python
def map_pairs(genotype_df, variant_df, expression_df, phenotype_pos_df, covariates_df=None,
              genes=None, pairs=None, maf_threshold=0.05, nperm=100000, window=1000000,
              cis_window_b=None, null='bucketed', beta_approx=True, perm_chunk_size=2000,
              seed=None, logger=None, verbose=True):
    """null: 'bucketed' one null per (A, marginal bucket) -- exact (default)
             'normal'   one null per A from norm.ppf(arange(1,n+1)/(n+1))  (trans.py:337)
             'exact'    one null per (A,B) by permuting B's own values -- reproduces reference
    """
```

### Setup (mirrors `cis.map_cis:633-678` so results are comparable)

```python
    assert expression_df.index.equals(phenotype_pos_df.index)              # trap T1
    assert (genotype_df.index == variant_df.index).all()                   # genotypeio.py:432
    assert covariates_df.index.equals(expression_df.columns)               # cis.py:645
    residualizer = cis.Residualizer(torch.tensor(covariates_df.values,
                                    dtype=torch.float32).to(device))       # core.py:62
    dof = n_samples - 2 - covariates_df.shape[1]; assert dof == residualizer.dof
    if seed is not None: np.random.seed(seed)                              # cis.py:674
    perm_ix = np.array([np.random.permutation(n_samples) for _ in range(nperm)])  # cis.py:675
```

### Marginal buckets + hoisted normalisation

```python
def _prep_rows(M_t, residualizer):
    """residualizer.transform (core.py:68) + center_normalize (core.py:135) -- i.e. the
    internals of core.calculate_corr (core.py:141-164), hoisted for reuse. r == Mn @ Nn.T"""
    R_t = residualizer.transform(M_t) if residualizer is not None else M_t
    return cis.center_normalize(R_t, dim=1), R_t.var(1)      # var matches core.py:152-153

def _marginal_buckets(expr_df, decimals=12):
    """Group genes by the multiset of their values (hash of the sorted vector).
    The permutation null depends on y only through this, so one null per bucket is EXACT."""
```

Per bucket, build the `nperm × N` null matrix as `sorted_values[perm_ix]` and pass it through
`_prep_rows` **once** — residualisation of the null is A-independent, so cache the normalised
result per bucket. Also cache `Bn_t, bvar_t = _prep_rows(B_t, residualizer)` for all M genes.
Log the bucket count and the max deviation from `norm.ppf(i/(n+1))`.

### Per-gene-A loop

```python
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, expr_df, pos_df, window=window)
    for _p, genotypes, genotype_range, gene_a in igc.generate_data(verbose=verbose):
        # block prep: order of operations copied verbatim from cis.py:686-703
        G_t = torch.tensor(genotypes, dtype=torch.float).to(device)   # copies -- see trap T2
        G_t = G_t[:, genotype_ix_t]
        cis.impute_mean(G_t)                                          # AFTER the sample subset
        if maf_threshold > 0: ... cis.calculate_maf(G_t) >= maf_threshold  # cis.py:691
        mono_t = (G_t == G_t[:, [0]]).all(1); ...                          # cis.py:698
        af_t, ma_samples_t, ma_count_t = cis.get_allele_stats(G_t)         # core.py:84
        Gn_t, gvar_t = _prep_rows(G_t, residualizer)

        r_t = torch.mm(Gn_t, Bn_t.t())      # V x M -- ONE GEMM for every B's nominal stats

        # partition B's into null groups: (leakage interval, marginal bucket)
        for (key_excl, bkey), cols in groups.items():
            keep = ...                       # None, or np.r_[0:lo, hi:V]
            r2_perm = _max_r2_perm(Gk_t, permn_cache[bkey], perm_chunk_size)   # cis.py:54-58
            bs1, bs2, true_dof, _ = cis.fit_beta_parameters(r2_perm, dof, return_minp=True)
            # min-p per B  (cis.py:60-63)
            r2k_t = torch.where(torch.isnan(rk_t.pow(2)), -1., rk_t.pow(2))   # cis.py:61
            ixc_t = r2k_t.argmax(0)                                           # cis.py:63
            ...
```

`_max_r2_perm` chunks over permutations and accumulates the max (the trick `trans.py:369-372`
uses over variant batches), so peak memory is `V × perm_chunk_size` instead of `V × nperm`.
Upstream `cis.py:54-55` does **not** chunk — that is why `nperm=1e5` OOMs in the reference but is
cheap here (trap T8).

### Per-pair scalars — vectorised `prepare_cis_output` (`cis.py:566-599`)

```python
        r2_nom   = r_nom * r_nom                                        # :566 (float32, see T3)
        slope    = r_nom * std_ratio                                    # :569  std_ratio = sqrt(bvar/gvar), :51
        tstat2   = dof * r2_nom / (1 - r2_nom)                          # :570
        slope_se = np.abs(slope) / np.sqrt(tstat2)                      # :571
        pval_nominal = cis.pval_from_corr(r2_nom.astype(np.float64), dof)   # :594 / core.py:339
        srt = np.sort(r2_perm)                                          # exact vectorisation of :567
        pval_perm = (nperm - np.searchsorted(srt, r2_nom, 'left') + 1) / (nperm + 1)
        pval_true_df = cis.pval_from_corr(r2_nom.astype(np.float64), true_dof)  # core.py:398
        pval_beta = stats.beta.cdf(pval_true_df, bs1, bs2)                     # core.py:399
```

`af`/`ma_samples`/`ma_count` come from `cis.get_allele_stats` (`core.py:84`) indexed by the lead
row — algebraically identical to `prepare_cis_output`'s scalar branch (`cis.py:573-580`).
Output columns match `map_cis` plus `gene_a`, `gene_b`, `same_chrom`, `leakage_removed`.

### Across-edge FDR

```python
def bh_edges(res_df, fdr=0.05, pval_col='pval_beta'):
    out['qval_bh'] = eigenmt.padjust_bh(res_df[pval_col].values)    # eigenmt.py:175, no R needed
    out['significant'] = out['qval_bh'] <= fdr
```

`post.calculate_qvalues` (`post.py:17`) also works as-is on this output (we emit `pval_beta`,
`pval_perm`, `beta_shape1/2`) and logs the Beta-vs-empirical correlation as a free calibration
diagnostic — but it needs R + `qvalue` + rpy2, and Storey's π₀ is unreliable on a few hundred
edges. BH is the default.

Optional: `eigenmt.compute_tests(G_t, var_thresh=0.99, variant_window=200)` (`eigenmt.py:85`)
runs standalone on the already-resident genotype block; expose it as `tests_emt` per A so
`pval_nominal * tests_emt` (the doc's Option B) can be compared against `pval_beta`.

---

## 4. Cis-leakage handling

Exclusion set for (A,B) = variants in A's window with `chrom==chrom(B)` and
`b_start - w_B <= pos <= b_end + w_B` (matching `trans._in_cis`, `trans.py:16-31`). The block is
position-sorted, so this is a **contiguous index interval** — two `np.searchsorted` calls.

The mask must be applied **before** both the min-p search and the null. Post-hoc filtering of the
output (e.g. `trans.filter_cis`) is wrong: removing the lead variant changes both the lead and
the null.

Because the null is keyed on the exclusion interval, per-pair masking costs almost nothing:
cross-chromosome pairs share one null; same-chromosome B's whose intervals clip the same variants
share a null; `num_var` is reported per group so the effective test count stays honest. Note the
bias direction if one *did* reuse a full-block null on a masked pair: a null over a superset is
stochastically larger in max-r², so `pval_beta` would be **conservative** — safe, but the keyed
version is exact.

`cis_window_b` removes the (A,A) diagonal entirely, so **the (A,A) validation test must run with
`cis_window_b=None`**.

---

## 5. Verification (`module-based/verify_module_edges.py`)

Fixtures from `example/data/` (445 samples, 26 covariates → dof = 417; genotypes are **chr18
only**, 367,786 variants over ~80 Mb ⇒ ~9,000 variants per ±1 Mb window, ~4,000 after MAF≥0.05;
`phenotype_pos_df` has `['chr','pos']`; 301 chr18 genes). Only chr18 genes can serve as **A**;
any gene can serve as **B**, which is what gives genuine trans pairs.

Run in this order — cheapest and most decisive first:

0. **Marginal diagnostic (no permutations).** `dev = |sort(y) - norm.ppf(i/(n+1))|.max(1)` per
   gene; report how many genes match exactly vs. have ties, and the bucket count for a candidate
   module. This is what justifies `null='bucketed'` over `null='normal'`, and takes seconds.
1. **(A,A) reproduces `cis.map_cis` bit-for-bit.** For ~25 chr18 genes, compare
   `map_pairs_reference(pairs=[(g,g)])` against `cis.map_cis` on those genes, same `seed`,
   `pd.testing.assert_frame_equal(..., check_exact=True)`. Exactness is legitimate:
   `permutation_ix_t` is drawn once from `seed` before the phenotype loop (`cis.py:672-675`),
   independent of phenotype count/order; `random_tiebreak=False` consumes no RNG;
   `fit_beta_parameters` is deterministic. This single test catches any A/B role transposition,
   index misalignment, or off-by-one in `pair_pos_df`. Only permitted deviation: `variant_id` on
   an exact r² tie (trap T4).
2. **Independent nominal cross-check.** For one A and ~6 B's (3 chr18, 3 chr7), run
   `trans.map_trans(..., return_sparse=False)` (`trans.py:53` — shares no code with
   `InputGeneratorCis`) and assert its minimum p over A's window variants, the argmin variant,
   `b` and `b_se` match our `pval_nominal`/`variant_id`/`slope`/`slope_se` (`rtol=1e-4`;
   differences are float32-vs-float64 squaring, `trans.py:141` vs `cis.py:566`). Repeat with
   `cis.calculate_association` (`cis.py:70`) at `maf_threshold=0`.
   Add a **transposition guard**: `(A,B)` and `(B,A)` must differ for cross-chromosome pairs.
3. **Fast vs reference**, ~20 A × ~29 B = 580 edges. Calibrate tolerances against the
   reference's *own* seed-to-seed spread rather than magic numbers:
   `median|Δlog10 pval_beta|(fast,ref1) <= 1.5 × median|Δ|(ref1,ref2)`, same for p99. Exact
   agreement required on `num_var`, `ma_samples`, `ma_count`, `start_distance`, `af`; `rtol=1e-5`
   on `pval_nominal`/`slope`/`slope_se`. Stratify by whether B's marginal is tied — `'bucketed'`
   should erase the difference that `'normal'` shows.
4. **Shortcut assumption, stratified.** For one A, compare `beta_shape1/2` and `true_df` from
   `null='bucketed'`, `'normal'`, and the per-B reference, over 6 untied and 6 tied B's.
   Predicted: untied B's agree under all three; tied B's show a *systematic* offset under
   `'normal'` that `'bucketed'` removes. If `'bucketed'` doesn't, the bucket key is wrong.
5. **Null calibration.** Permute the **genotype** sample labels (not expression — expression must
   stay aligned with covariates), so the Beta fit is not trained on the same null. Sample **one
   edge per A** across ~20 independent shuffles (edges sharing an A are dependent through the
   shared Beta fit) → `stats.kstest(x, 'uniform')` should not reject at 0.01. Also assert
   `pval_nominal` is *strongly* non-uniform (KS p < 1e-10) — this catches the degenerate case
   where the min-p search collapses to one variant, under which `pval_beta` would still look
   uniform.
6. **End-to-end smoke run.** ~30 module genes → `build_pairs` → `map_pairs` → `bh_edges`; print
   edges at q < 0.05 and the count excluded for cis leakage.

---

## 6. Cost

The GEMM is **not** the bottleneck — `core.fit_beta_parameters` is. It runs Newton root-finding
plus two Nelder-Mead fits over the whole `r2_perm` vector on the CPU (`core.py:358-388`), roughly
40 ms per call at nperm=1e4 (≈0.4 s at 1e5). So cost ≈ (number of Beta fits) × that, and the
whole design reduces to *minimising the number of Beta fits*: **M for the fast path, M(M−1) for
the reference** — a factor of M−1.

| | Beta fits | M=100 (9.9k edges) | M=300 (90k) | M=1000 (1M) |
|---|---|---|---|---|
| fast, nperm=1e4 | M | ~10 s | ~30 s | ~2 min |
| fast, nperm=1e5 | M | ~1 min | ~3 min | ~10 min |
| reference, nperm=1e4 | M(M−1) | ~1.5 h | ~12 h | infeasible |

(CPU-only, 12 cores; the GEMM adds ~0.5 s/gene-A at nperm=1e4, ~5 s at 1e5. A GPU cuts the GEMM
to milliseconds but not the Beta fits, so the fast path is CPU-bound either way — meaning **this
GPU-less machine is fine for the fast path**, and further speedup means a `ProcessPoolExecutor`
over `r2_perm` arrays, not more GPU.)

**Raise `nperm` to 1e5 in the fast path.** Under a shared null, the Monte-Carlo error in
`(beta_shape1, beta_shape2, true_df)` is *perfectly correlated* across all M−1 edges of a gene A
— roughly 4% on `pval_beta` at p=1e-3 and 8–10% at p=1e-5 at nperm=1e4 — which inflates
out-degree variance and manufactures spurious "hub" genes after BH (trap T7). It is cheap to fix
here and impossible in the reference (which OOMs, trap T8).

**Memory.** Fast: `Gn` V×N (~9 MB) + `Bn` M×N + one normalised null per bucket (18 MB at
nperm=1e4, 178 MB at 1e5) + a V×`perm_chunk_size` chunk → well under 1 GB, independent of nperm;
host-side `perm_ix` is 36 MB at 1e4 / 356 MB at 1e5 as int64 (use int32), and the output frame is
~300 MB at E=1e6 → write per-A parquet. Reference: `cis.py:54-55` materialises ~3 copies of a
V×nperm float32 matrix (~1.1 GB at V=9,000, nperm=1e4) plus the E×N pseudo-phenotype matrix
(1.8 GB at M=1000 — build it float32).

---

## 7. Correctness traps

- **T1 — silent index misalignment (highest consequence).** `map_cis` never asserts
  `phenotype_df.index.equals(phenotype_pos_df.index)` (contrast `cis.py:819`), and
  `InputGeneratorCis` does `phenotype_df[m]` with a boolean mask (`genotypeio.py:443-447`) —
  a misordered mask silently reindexes on modern pandas, pairing A's window with the wrong B and
  producing plausible-looking wrong answers. Assert equality in both entry points.
- **T2 — in-place mutation.** `cis.impute_mean` mutates its argument (`core.py:124-132`) and
  `genotype_df.values[r0:r1+1]` is a numpy **view**. `torch.tensor(...)` copies, so upstream is
  safe — but "optimising" to `torch.from_numpy(...)` would corrupt `genotype_df` for every later
  gene. Also impute *after* the sample subset (`cis.py:686-688`), since the mean is over selected
  samples.
- **T3 — float32 `r2_perm`.** `cis.py:719` passes float32 into `calculate_beta_approx_pval` while
  `trans.py:424` casts to float64. The difference is ~1e-6 relative on `beta_shape2` — negligible
  except when `r2_perm → 1` (cancellation in `dof·r²/(1−r²)`, `core.py:340`). Keep float32 so the
  fast path stays comparable to the reference; expose a dtype switch and document it.
- **T4 — argmax ties are not reproducible on CUDA.** `cis.py:63` and our `argmax(0)` may pick
  different tied variants across devices; perfect LD in a 2 Mb window makes this common. Compare
  `variant_id` leniently and assert `pval_nominal` agreement instead. Don't use
  `random_tiebreak=True` in comparisons.
- **T5 — `astype(output_dtype_dict)` raises on absent keys.** Filter the dict to columns actually
  present. Related: `start_distance`/`end_distance` are `np.int32` (`core.py:30-31`) — safe here
  because they are relative to **A** — but any distance-to-**B** column must be float or nullable
  `Int32`, since it is undefined for cross-chromosome pairs.
- **T6 — dropped pairs are silent.** `InputGeneratorCis` drops phenotypes at
  `genotypeio.py:445, 450-454, 467-470` and `map_cis` skips at `cis.py:705-707`. Always reconcile
  the output against the requested pair list and report why each missing pair vanished. In the
  fast path, a gene unusable as **A** (no in-window variants) is still usable as **B**, so build
  the B-side tensor from `expr_df`, **not** from `igc.phenotype_df`.
- **T7 — shared-null MC error is correlated across an A's edges.** Not a bug but a statistical
  hazard (hub artifacts). Mitigate with large `nperm`; report `beta_shape2`/`true_df` per A so it
  is auditable.
- **T8 — raising `nperm` in the reference OOMs** (see §6 memory).
- **T9 — constant / near-constant B.** `center_normalize` divides by the row norm
  (`core.py:138`) → NaN for a constant row. `map_cis` is protected by `igc` dropping constant
  phenotypes (`genotypeio.py:450`) plus the NaN filters at `cis.py:55, 61`. In the fast path B's
  never pass through `igc`, so reproduce both explicitly.
- **T10 — `paired_covariate_df` is incompatible** with any shared null (per-phenotype
  residualizer and `dof`, `cis.py:653-658, 710-716`). Raise `NotImplementedError`.
- **T11 — beta-fit init inconsistency upstream** (`dof*0.25` grouped at `cis.py:621` vs `dof`
  ungrouped at `cis.py:727`): the Newton solve converges to the same answer either way. Use
  `dof`, matching the ungrouped path being replicated.
- **T12 — the example genotypes are chr18-only**, so `InputGeneratorCis` prints a "dropping N
  phenotypes on chrs. without genotypes" warning for every non-chr18 gene on the A side
  (`genotypeio.py:445`). Expected, not a bug.
- **T13 — README's `trans.filter_cis` example is stale** (README.md:173 passes `.T.to_dict()`;
  the function does that internally, `trans.py:42`). Don't copy it.

## Interpretation caveat to record in the output

Per §5 of `module-based-approach.md`, an edge means "genetic variation near A is associated with
expression of B", not "A regulates B". Also, because the module was *selected* for GWAS/TWAS
enrichment, edge-FDR is conditional on that selection. Keep `gene_a`, `gene_b`, `variant_id`,
`start_distance`, `slope`, `pval_beta`, `qval_bh` in the output so the mediation step
(SNP → A expression → B expression) can be layered on later without re-running the scan.
