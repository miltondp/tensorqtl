"""Edge-level (locus A -> expression B) QTL mapping for gene modules.

Implements step 2 ("within-edge multiple testing") and step 3 ("across-edge FDR") of
``module-based-approach.md``: for every ordered gene pair (A, B) in a module, test all
variants in A's cis-window against the expression of gene B, and reduce that to a single
LD-aware, calibrated edge-level p-value via the min-p permutation + Beta-approximation
scheme that tensorQTL uses for cis-QTL mapping.

Because module genes are usually far apart (often on different chromosomes), these are
trans-associations -- but the *statistic* is the cis one: min-p over the variants of a
single locus, calibrated by permutation. ``trans.map_trans`` cannot be used because it
returns nominal p-values only.

Two entry points:

``map_pairs_reference``
    Ground truth. Builds a *pseudo-phenotype* matrix (row values = expression of B, row
    position = TSS of A) and calls ``cis.map_cis`` unmodified. ``InputGeneratorCis``
    selects variants purely from the position frame and never inspects which gene a
    phenotype row belongs to, so this tests A's window against B's expression. O(M^2)
    permutations -- a validation tool, not production.

``map_pairs``
    Production. Iterates once per gene A, and fits the permutation null once per
    (gene A, marginal bucket) instead of once per pair. This is exact, not an
    approximation: the null distribution of ``max_v r^2(g_v, pi(y))`` depends on the
    phenotype ``y`` only through its multiset of values, so phenotypes with identical
    sorted value vectors share an identical null. A properly inverse-normal-transformed
    module collapses to a single bucket -> O(M) instead of O(M^2).

    (Upstream's ``trans.map_permutations`` uses the special case of this where every
    phenotype is assumed to equal ``norm.ppf(arange(1,n+1)/(n+1))``. That assumption
    fails for genes with tied values, which is why the bucketing is keyed on the actual
    sorted values instead.)

    The cost is therefore ``n_loci x n_buckets`` permutation nulls, not ``n_pairs``, and the
    speedup is driven by the *number of distinct marginals in the module*, not by its size.
    Measured on the bundled data (445 samples, ~4k-8k variants per window, nperm=1e4, 12-core
    CPU): 0.70 s per locus with 1 bucket vs 5.87 s with 8 buckets, against 0.71 s **per pair**
    for the reference. ``map_pairs`` logs how many nulls it fitted, so this is visible per run.

    To get to one bucket, the module's expression must be rank-normalized with *tie-breaking*
    -- a plain inverse-normal transform is not enough, because average ranks for tied values
    produce a distinct quantile vector per gene (verified: 8 buckets before and after). Genes
    with many tied values are low-count genes, which are usually filtered out before building
    a co-expression module in the first place.

Nothing under ``tensorqtl/`` is modified by this module.

A calibration caveat inherited from upstream
--------------------------------------------
Under a genotype-label-shuffled null on the bundled GEUVADIS data (445 samples), edge
p-values are exactly uniform when no covariates are fitted, but drift away from uniform --
usually conservatively -- as the covariate count grows: mean edge p-value 0.51, 0.51, 0.53,
0.59 with 0, 5, 13 and 26 covariates respectively. ``cis.map_cis`` shows the same drift on
the same data to within Monte-Carlo noise (paired spearman 0.996), so this is a property of
the FastQTL-style scheme of permuting the *raw* phenotype and residualizing afterwards
(cis.py:47), not of this module. It matters when covariates are numerous relative to the
sample size; see test 5 of ``verify_module_edges.py``.
"""

import re
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

# Core primitives are reached through `cis` on purpose: cis.py does `from core import *`,
# so `cis.calculate_corr` etc. are the exact same function objects `map_cis` calls. Going
# through `tensorqtl.core` instead would import a *second* copy of the module, because the
# package uses flat `sys.path`-based imports internally (see CLAUDE.md).
from tensorqtl import cis, eigenmt, genotypeio

PAIR_SEP = '|'

#: Columns emitted by ``cis.prepare_cis_output`` (cis.py:582-599), in order.
CIS_OUTPUT_COLUMNS = [
    'num_var', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df', 'variant_id',
    'start_distance', 'end_distance', 'ma_samples', 'ma_count', 'af', 'pval_nominal',
    'slope', 'slope_se', 'pval_perm', 'pval_beta',
]


# ---------------------------------------------------------------------------------------
# position helpers
# ---------------------------------------------------------------------------------------

def _get_start_end(pos_df):
    """Return (chrom, start, end) arrays from a phenotype position frame.

    ``core.read_phenotype_bed`` (core.py:406) yields either ['chr', 'pos'] (when
    start + 1 == end for every feature) or ['chr', 'start', 'end']. Both are accepted by
    ``InputGeneratorCis``, so both must be accepted here. Matches ``trans._in_cis``
    semantics (trans.py:20-25).
    """
    chrom = pos_df['chr'].values
    if 'pos' in pos_df:
        start = pos_df['pos'].values
        end = start
    else:
        start = pos_df['start'].values
        end = pos_df['end'].values
    return chrom, np.asarray(start), np.asarray(end)


def build_pairs(module_genes, phenotype_pos_df, min_pair_distance=5000000,
                include_self=False, pairs=None):
    """Enumerate ordered gene pairs, flagging those at risk of cis-leakage.

    An edge A->B is only interpretable as trans if A's cis-window cannot reach B's own
    cis-regulatory region; otherwise the "trans" signal is just B's cis-eQTL rediscovered
    through a nearby window (section 4B of module-based-approach.md).

    Args:
        module_genes: gene IDs in the module (must be present in phenotype_pos_df).
        phenotype_pos_df: position frame, ['chr','pos'] or ['chr','start','end'].
        min_pair_distance: exclude same-chromosome pairs closer than this (bp).
        include_self: keep the (A,A) diagonal (used by the verification suite).
        pairs: optional explicit list of (gene_a, gene_b) tuples instead of the full grid.

    Returns:
        DataFrame ['gene_a','gene_b','same_chr','gene_distance','excluded'].
        ``gene_distance`` is NaN for cross-chromosome pairs.
    """
    module_genes = list(module_genes)
    missing = set(module_genes) - set(phenotype_pos_df.index)
    if missing:
        raise ValueError(f"{len(missing)} genes not in phenotype_pos_df, e.g. {sorted(missing)[:3]}")

    if pairs is None:
        pairs = [(a, b) for a in module_genes for b in module_genes
                 if include_self or a != b]
    pair_df = pd.DataFrame(list(pairs), columns=['gene_a', 'gene_b'])
    if pair_df.duplicated().any():
        raise ValueError('duplicate (gene_a, gene_b) pairs')

    chrom, start, end = _get_start_end(phenotype_pos_df)
    ix = {g: i for i, g in enumerate(phenotype_pos_df.index)}
    ia = pair_df['gene_a'].map(ix).values
    ib = pair_df['gene_b'].map(ix).values

    pair_df['same_chr'] = chrom[ia] == chrom[ib]
    # distance between the two gene intervals (0 if they overlap)
    dist = np.maximum(np.maximum(start[ia] - end[ib], start[ib] - end[ia]), 0).astype(float)
    dist[~pair_df['same_chr'].values] = np.nan
    pair_df['gene_distance'] = dist
    pair_df['excluded'] = pair_df['same_chr'].values & (dist < min_pair_distance)
    return pair_df


def _annotate_leakage(res_df, variant_df, phenotype_pos_df, window):
    """Add ``lead_variant_in_cis_of_b``: is the lead variant within +-window of gene B?

    Vectorised equivalent of ``trans._in_cis(chrom, pos, gene_b, pos_dict, window)``
    (trans.py:16-31), used as a safety net even when nearby pairs are already excluded.
    """
    chrom, start, end = _get_start_end(phenotype_pos_df)
    gix = {g: i for i, g in enumerate(phenotype_pos_df.index)}
    ib = res_df['gene_b'].map(gix).values
    vchrom = variant_df['chrom'].reindex(res_df['variant_id']).values
    vpos = variant_df['pos'].reindex(res_df['variant_id']).values
    res_df['lead_variant_in_cis_of_b'] = (
        (vchrom == chrom[ib]) & (vpos >= start[ib] - window) & (vpos <= end[ib] + window)
    )
    return res_df


# ---------------------------------------------------------------------------------------
# reference implementation: pseudo-phenotypes + unmodified cis.map_cis
# ---------------------------------------------------------------------------------------

def build_pair_frames(expression_df, phenotype_pos_df, pairs, sep=PAIR_SEP, dtype=np.float32):
    """Build the pseudo-phenotype / pseudo-position frames that drive ``cis.map_cis``.

        pair_expr_df.loc[f"{A}{sep}{B}"] == expression of gene B
        pair_pos_df.loc[f"{A}{sep}{B}"]  == position of gene A     <-- the whole trick

    ``InputGeneratorCis`` reads variant windows out of the position frame only
    (``get_cis_ranges``, genotypeio.py:380), so the pair is tested as
    "variants near A" vs "expression of B".

    Returns (pair_expr_df, pair_pos_df, pair_df) where pair_df is indexed by pair_id.
    """
    if not expression_df.index.equals(phenotype_pos_df.index):
        # cis.map_cis never checks this and InputGeneratorCis masks the two frames
        # independently (genotypeio.py:443-447); a mismatch silently pairs A's window
        # with the wrong gene B.
        raise ValueError('expression_df and phenotype_pos_df must share index and order')
    if expression_df.index.str.contains(re.escape(sep)).any():
        raise ValueError(f'gene IDs must not contain the pair separator {sep!r}')

    pair_df = pd.DataFrame(list(pairs), columns=['gene_a', 'gene_b'])
    unknown = (set(pair_df['gene_a']) | set(pair_df['gene_b'])) - set(expression_df.index)
    if unknown:
        raise ValueError(f'{len(unknown)} genes not in expression_df, e.g. {sorted(unknown)[:3]}')
    pair_df['pair_id'] = pair_df['gene_a'] + sep + pair_df['gene_b']
    if pair_df['pair_id'].duplicated().any():
        raise ValueError('duplicate (gene_a, gene_b) pairs')  # genotypeio.py:433

    # visit A's genotype blocks in genomic order
    a_pos = phenotype_pos_df.loc[pair_df['gene_a']].reset_index(drop=True)
    poscol = 'pos' if 'pos' in a_pos else 'start'
    order = np.lexsort((pair_df['gene_b'].values, pair_df['gene_a'].values,
                        a_pos[poscol].values, a_pos['chr'].values))
    pair_df = pair_df.iloc[order].reset_index(drop=True)

    pair_expr_df = pd.DataFrame(
        expression_df.loc[pair_df['gene_b']].values.astype(dtype),
        index=pair_df['pair_id'].values, columns=expression_df.columns)
    pair_pos_df = phenotype_pos_df.loc[pair_df['gene_a']].copy()
    pair_pos_df.index = pair_df['pair_id'].values

    assert pair_expr_df.index.equals(pair_pos_df.index)
    return pair_expr_df, pair_pos_df, pair_df.set_index('pair_id')


def map_pairs_reference(genotype_df, variant_df, expression_df, phenotype_pos_df,
                        covariates_df=None, pairs=None, window=1000000,
                        annotate_leakage_window=1000000, **kwargs):
    """Edge-level p-values via pseudo-phenotypes + unmodified ``cis.map_cis`` (cis.py:627).

    Ground truth for ``map_pairs``. Cost is O(n_pairs) permutation runs *and* O(n_pairs)
    ``fit_beta_parameters`` calls, so this is only practical for a few hundred pairs.

    ``pairs`` is a list of (gene_a, gene_b) tuples. Extra kwargs go to ``cis.map_cis``
    (``nperm``, ``maf_threshold``, ``seed``, ``beta_approx``, ``random_tiebreak``, ...).
    """
    if pairs is None:
        raise ValueError('pairs is required')
    pair_expr_df, pair_pos_df, pair_df = build_pair_frames(
        expression_df, phenotype_pos_df, pairs)

    kwargs.setdefault('warn_monomorphic', False)  # cis.py:702 would print once per pair
    res_df = cis.map_cis(genotype_df, variant_df, pair_expr_df, pair_pos_df,
                         covariates_df=covariates_df, window=window, **kwargs)

    # InputGeneratorCis drops phenotypes silently (genotypeio.py:445, 450-454, 467-470)
    # and map_cis skips at cis.py:705-707 -- reconcile rather than assume 1:1 rows.
    dropped = pair_df.index.difference(res_df.index)
    if len(dropped) > 0:
        print(f'    ** {len(dropped)} pairs missing from output '
              f'(no in-window variants / constant phenotype / contig without genotypes), '
              f'e.g. {list(dropped[:3])}')
    out_df = pair_df[['gene_a', 'gene_b']].join(res_df, how='inner')
    out_df.index.name = 'pair_id'
    if annotate_leakage_window is not None:
        _annotate_leakage(out_df, variant_df, phenotype_pos_df, annotate_leakage_window)
    return out_df


# ---------------------------------------------------------------------------------------
# fast implementation
# ---------------------------------------------------------------------------------------

def _prep_rows(M_t, residualizer):
    """Residualize + center/normalize rows, i.e. the internals of ``core.calculate_corr``.

    ``calculate_corr`` (core.py:141-164) residualizes both arguments, records their
    variance, then ``center_normalize``s them and does one matmul. Splitting it lets the
    expensive per-row work be hoisted out of the pair loop; afterwards
    ``r == Mn_t @ Nn_t.T`` exactly as ``calculate_corr`` would return.

    Returns (normalized rows, per-row variance of the *residual* (ddof=1)).
    """
    R_t = residualizer.transform(M_t) if residualizer is not None else M_t  # core.py:68
    var_t = R_t.var(1)                                                     # core.py:153
    return cis.center_normalize(R_t, dim=1), var_t                         # core.py:135


def _max_r2_perm(Gn_t, null_n_t, chunk_size=2000):
    """max over variants of r^2, per null phenotype.

    Same statistic as ``cis.calculate_cis_permutations`` (cis.py:54-58) and
    ``trans.map_permutations`` (trans.py:369-372), but chunked over permutations so peak
    memory is n_variants x chunk_size rather than n_variants x nperm. Upstream does not
    chunk, which is what makes large ``nperm`` infeasible in ``map_cis``.

    ``Gn_t`` and ``null_n_t`` must already be normalized by ``_prep_rows``.
    """
    # Rows that residualize to zero variance give NaN correlations for *every*
    # permutation; cis.py:55 drops them before taking the max.
    valid_t = ~torch.isnan(Gn_t).any(1)
    if not bool(valid_t.all()):
        Gn_t = Gn_t[valid_t]
    if Gn_t.shape[0] == 0:
        raise ValueError('All correlations resulted in NaN. Please check phenotype values.')

    nperm = null_n_t.shape[0]
    out_t = torch.empty(nperm, dtype=torch.float32, device=Gn_t.device)
    for s in range(0, nperm, chunk_size):
        r2_t = torch.mm(Gn_t, null_n_t[s:s + chunk_size].t()).pow(2)
        out_t[s:s + chunk_size] = r2_t.max(0)[0]
    return out_t.cpu().numpy()


def _marginal_buckets(values, decimals=12):
    """Group phenotypes by the multiset of their values.

    The permutation null of ``max_v r^2(g_v, pi(y))`` is a function of ``y`` only through
    its multiset of values, so all phenotypes in a bucket share an *identical* null and
    one permutation run per bucket is exact.

    Args:
        values: (n_phenotypes, n_samples) array.
        decimals: rounding applied before hashing, to absorb float noise.

    Returns:
        (bucket_key per phenotype, {bucket_key: sorted values (float64)}).
    """
    sorted_vals = np.sort(np.asarray(values, dtype=np.float64), axis=1)
    keys = np.empty(sorted_vals.shape[0], dtype=object)
    reps = {}
    for i, row in enumerate(np.round(sorted_vals, decimals)):
        k = row.tobytes()
        keys[i] = k
        if k not in reps:
            reps[k] = sorted_vals[i]
    return keys, reps


class _NullCache:
    """Lazily build (and optionally cache) normalized permutation-null matrices.

    A null matrix is ``sorted_values[perm_ix]`` -- nperm x n_samples -- pushed through
    ``_prep_rows``. Residualization does not depend on gene A, so caching is a pure win;
    but each entry costs nperm * n_samples * 4 bytes on the device, so the number kept is
    capped and the rest recomputed on demand.
    """

    def __init__(self, perm_ix, residualizer, device, max_cached=8):
        self.perm_ix = perm_ix
        self.residualizer = residualizer
        self.device = device
        self.max_cached = max_cached
        self._cache = {}

    def get(self, key, sorted_vals):
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        P_t = torch.tensor(sorted_vals.astype(np.float32)[self.perm_ix],
                           dtype=torch.float32).to(self.device)
        Pn_t, _ = _prep_rows(P_t, self.residualizer)
        del P_t
        if len(self._cache) < self.max_cached:
            self._cache[key] = Pn_t
        return Pn_t


def map_pairs(genotype_df, variant_df, expression_df, phenotype_pos_df, covariates_df=None,
              pairs=None, genes=None, window=1000000, nperm=100000, maf_threshold=0.05,
              null='bucketed', beta_approx=True, cis_window_b=None, perm_chunk_size=2000,
              max_cached_nulls=8, run_eigenmt=False, annotate_leakage_window=1000000,
              seed=None, random_tiebreak=False, paired_covariate_df=None,
              logger=None, verbose=True, warn_monomorphic=False):
    """Edge-level p-values for many gene pairs, fitting one permutation null per locus.

    One iteration per gene A: A's genotype block is loaded once, every partner B's nominal
    statistics come from a single matmul, and the permutation null / Beta fit is shared by
    all B's that have the same value multiset (see ``_marginal_buckets``).

    Args:
        genotype_df: variants x samples; index must match ``variant_df.index``.
        variant_df: index = variant_id, columns ['chrom', 'pos'], sorted by pos within chrom.
        expression_df: genes x samples.
        phenotype_pos_df: positions for the genes in ``expression_df``, same index/order.
        covariates_df: samples x covariates.
        pairs: list of (gene_a, gene_b). Defaults to all ordered pairs among ``genes``.
        genes: module genes; defaults to ``expression_df.index``.
        null: 'bucketed' -- one null per (A, value-multiset bucket); exact (default).
              'normal'   -- one null per A from norm.ppf(arange(1,n+1)/(n+1)), i.e. the
                            assumption ``trans.map_permutations`` makes (trans.py:337).
                            Only correct if every phenotype really has that marginal.
              'exact'    -- one null per (A, B), permuting B's own values. Reproduces
                            ``map_pairs_reference``; O(n_pairs).
        cis_window_b: if set, drop variants within this many bp of gene B before both the
            min-p search and the permutation null (only bites when A and B share a
            chromosome). Note this removes the (A,A) diagonal entirely.
        nperm: permutations per null. Large values are cheap here (one fit per locus) and
            reduce the Monte-Carlo error that a shared null correlates across all of A's
            edges.

    Returns:
        DataFrame indexed by ``f"{gene_a}|{gene_b}"`` with ``cis.map_cis``'s columns plus
        gene_a, gene_b, same_chrom, leakage_removed (and tests_emt if run_eigenmt).
    """
    if null not in ('bucketed', 'normal', 'exact'):
        raise ValueError(f"null must be 'bucketed', 'normal' or 'exact', got {null!r}")
    if paired_covariate_df is not None:
        # A phenotype-specific covariate changes both the residualizer and dof per gene B
        # (cis.py:653-658, 710-716), so no null can be shared across B.
        raise NotImplementedError('paired_covariate_df is incompatible with a shared null; '
                                  'use map_pairs_reference')
    if random_tiebreak:
        # cis.py:64-66 breaks ties per phenotype with an RNG draw; vectorizing that over B
        # would not reproduce map_cis's draw order anyway, and it is only reachable on exact
        # r^2 ties. Comparisons against the reference must use random_tiebreak=False.
        raise NotImplementedError('random_tiebreak is not supported; see map_pairs_reference')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logger if logger is not None else cis.SimpleLogger()

    # ---- inputs -----------------------------------------------------------------------
    if not expression_df.index.equals(phenotype_pos_df.index):
        raise ValueError('expression_df and phenotype_pos_df must share index and order')
    assert (genotype_df.index == variant_df.index).all(), \
        'genotype_df and variant_df indexes do not match'  # genotypeio.py:432

    genes = list(expression_df.index) if genes is None else list(genes)
    if len(set(genes)) != len(genes):
        raise ValueError('duplicate entries in `genes`')
    expr_df = expression_df.loc[genes]
    pos_df = phenotype_pos_df.loc[genes]
    n_genes, n_samples = expr_df.shape
    gene_ix = {g: i for i, g in enumerate(genes)}

    if pairs is None:
        pairs = [(a, b) for a in genes for b in genes if a != b]
    pairs = list(pairs)
    if len(set(pairs)) != len(pairs):
        raise ValueError('duplicate (gene_a, gene_b) pairs')  # would collide in the output index
    b_ix_by_a = {}
    for a, b in pairs:
        if a not in gene_ix or b not in gene_ix:
            raise ValueError(f'pair ({a}, {b}) contains a gene outside `genes`')
        b_ix_by_a.setdefault(a, []).append(gene_ix[b])
    b_ix_by_a = {a: np.array(v) for a, v in b_ix_by_a.items()}
    n_pairs = sum(len(v) for v in b_ix_by_a.values())

    logger.write('edge-QTL mapping: locus A -> expression B')
    logger.write(f'  * {n_samples} samples')
    logger.write(f'  * {n_genes} genes, {n_pairs} ordered pairs')
    logger.write(f'  * cis-window (around A): +-{window:,}')
    logger.write(f'  * {nperm} permutations, null={null!r}')
    if cis_window_b is not None:
        logger.write(f'  * excluding variants within +-{cis_window_b:,} of gene B')

    # ---- residualizer / dof -----------------------------------------------------------
    if covariates_df is not None:
        assert covariates_df.index.equals(expr_df.columns), \
            'Sample names in phenotype matrix columns and covariate matrix rows do not match!'
        # NB: upstream writes `assert ~(...)` here (cis.py:646), which never fires -- ~True
        # is -2, which is truthy. Use `not`.
        assert not covariates_df.isnull().any().any(), 'Missing or null values in covariates'
        residualizer = cis.Residualizer(
            torch.tensor(covariates_df.values, dtype=torch.float32).to(device))  # core.py:62
        dof = n_samples - 2 - covariates_df.shape[1]                             # cis.py:649
        assert dof == residualizer.dof                                           # core.py:66
        logger.write(f'  * {covariates_df.shape[1]} covariates (dof = {dof})')
    else:
        residualizer = None
        dof = n_samples - 2

    # ---- permutation indices (same construction/seeding as cis.map_cis) ---------------
    if seed is not None:
        logger.write(f'  * using seed {seed}')
        np.random.seed(seed)                                                     # cis.py:674
    perm_ix = np.array([np.random.permutation(n_samples) for _ in range(nperm)],
                       dtype=np.int32)                                           # cis.py:675

    # ---- marginal buckets -------------------------------------------------------------
    q_normal = stats.norm.ppf(np.arange(1, n_samples + 1) / (n_samples + 1))     # trans.py:337
    bucket_key, bucket_vals = _marginal_buckets(expr_df.values)
    max_dev = max(np.abs(v - q_normal).max() for v in bucket_vals.values())
    logger.write(f'  * {len(bucket_vals)} distinct phenotype marginals among {n_genes} genes '
                 f'(max deviation from norm.ppf(i/(n+1)): {max_dev:.3g})')
    if null == 'normal':
        if len(bucket_vals) > 1:
            logger.write(f'    ** WARNING: null=\'normal\' assumes a single marginal, but '
                         f'{len(bucket_vals)} were found; edge p-values for genes with '
                         f'tied values will be miscalibrated. Use null=\'bucketed\'.')
        bucket_key = np.array([b'_normal'] * n_genes, dtype=object)
        bucket_vals = {b'_normal': q_normal}
    elif null == 'exact':
        sorted_vals = np.sort(expr_df.values.astype(np.float64), axis=1)
        bucket_key = np.array([f'_g{i}'.encode() for i in range(n_genes)], dtype=object)
        bucket_vals = {bucket_key[i]: sorted_vals[i] for i in range(n_genes)}

    null_cache = _NullCache(perm_ix, residualizer, device,
                            max_cached=1 if null == 'exact' else max_cached_nulls)

    # ---- B side, computed once --------------------------------------------------------
    B_t = torch.tensor(expr_df.values, dtype=torch.float32).to(device)
    Bn_t, bvar_t = _prep_rows(B_t, residualizer)
    del B_t
    # Constant phenotypes normalize to 0/0 = NaN (core.py:138). map_cis is shielded by
    # InputGeneratorCis dropping them (genotypeio.py:450); B's never pass through it.
    b_const = np.all(expr_df.values == expr_df.values[:, [0]], axis=1)
    if b_const.any():
        logger.write(f'    ** {b_const.sum()} constant genes cannot be used as gene B')

    genotype_ix = np.array([genotype_df.columns.tolist().index(i)
                            for i in expr_df.columns])                           # cis.py:666
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    b_chrom, b_start, b_end = _get_start_end(pos_df)
    vpos_all = variant_df['pos'].values
    vid_all = variant_df.index.values

    # ---- drive the loop with the real genes, one iteration per gene A -----------------
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, expr_df, pos_df, window=window)
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')
    # A gene can be unusable as A (no in-window variants, or a contig without genotypes --
    # genotypeio.py:445, 467-470) while remaining perfectly usable as B, which is why the
    # B-side tensor is built from expr_df above and not from igc.phenotype_df.
    usable_a = set(igc.phenotype_df.index) & set(b_ix_by_a)
    unusable_a = sorted(set(b_ix_by_a) - usable_a)
    n_drop_a = sum(len(b_ix_by_a[a]) for a in unusable_a)
    logger.write(f'  * {len(usable_a)} of {len(b_ix_by_a)} genes usable as gene A')
    if unusable_a:
        logger.write(f'    ** {n_drop_a} pairs dropped: {len(unusable_a)} genes cannot serve '
                     f'as gene A (no variants in window / contig without genotypes), '
                     f'e.g. {unusable_a[:3]}')

    out = []
    n_null_fits = 0  # the cost driver: ~one per (gene A, marginal bucket)
    start_time = time.time()
    for phenotype, genotypes, genotype_range, gene_a in igc.generate_data(verbose=verbose):
        b_ix = b_ix_by_a.get(gene_a)
        if b_ix is None or len(b_ix) == 0:
            continue

        # --- genotype block; order of operations copied from cis.py:686-703 ---
        # torch.tensor() copies: `genotypes` is a view into genotype_df.values and
        # impute_mean mutates in place (core.py:124).
        G_t = torch.tensor(genotypes, dtype=torch.float).to(device)
        G_t = G_t[:, genotype_ix_t]
        cis.impute_mean(G_t)  # after the sample subset: the mean is over selected samples
        if maf_threshold > 0:
            mask_t = cis.calculate_maf(G_t) >= maf_threshold                     # cis.py:691
            G_t = G_t[mask_t]
            genotype_range = genotype_range[mask_t.cpu().numpy().astype(bool)]
        mono_t = (G_t == G_t[:, [0]]).all(1)                                     # cis.py:698
        if bool(mono_t.any()):
            G_t = G_t[~mono_t]
            genotype_range = genotype_range[~mono_t.cpu().numpy()]
            if warn_monomorphic:
                logger.write(f'    * WARNING: excluding {int(mono_t.sum())} monomorphic variants')
        if G_t.shape[0] == 0:
            logger.write(f'WARNING: skipping {gene_a} (no valid variants)')      # cis.py:706
            continue

        af_t, ma_samples_t, ma_count_t = cis.get_allele_stats(G_t)               # core.py:84
        # eigenMT runs on the raw imputed genotypes, as run_eigenmt does (eigenmt.py:161),
        # i.e. it is not covariate-adjusted -- so compute it before residualizing.
        m_eff = eigenmt.compute_tests(G_t) if run_eigenmt else None              # eigenmt.py:85
        Gn_t, gvar_t = _prep_rows(G_t, residualizer)
        del G_t
        n_var_full = Gn_t.shape[0]
        vpos = vpos_all[genotype_range]  # sorted within the block

        # one matmul for every B's nominal correlations
        r_all_t = torch.mm(Gn_t, Bn_t.t())  # n_variants x n_genes

        # --- partition B's into null groups: (variant-exclusion interval, marginal) ---
        a_chrom = pos_df['chr'].loc[gene_a]
        groups = {}
        for j in b_ix:
            if b_const[j]:
                continue
            key_excl = None
            if cis_window_b is not None and b_chrom[j] == a_chrom:
                lo = int(np.searchsorted(vpos, b_start[j] - cis_window_b, 'left'))
                hi = int(np.searchsorted(vpos, b_end[j] + cis_window_b, 'right'))
                if lo != hi:
                    key_excl = (lo, hi)
            groups.setdefault((key_excl, bucket_key[j]), []).append(j)

        for (key_excl, bkey), cols in groups.items():
            cols = np.asarray(cols)
            if key_excl is None:
                keep = None
                Gk_t, rk_t, num_var = Gn_t, r_all_t[:, cols], n_var_full
            else:
                lo, hi = key_excl
                keep = np.r_[np.arange(0, lo), np.arange(hi, n_var_full)]
                num_var = len(keep)
                if num_var == 0:
                    logger.write(f'WARNING: skipping {gene_a} -> {len(cols)} pairs '
                                 f'(all variants removed by cis_window_b)')
                    continue
                keep_t = torch.as_tensor(keep, device=device)
                Gk_t, rk_t = Gn_t[keep_t], r_all_t[keep_t][:, cols]

            # (a) permutation null over exactly the variants that were searched
            null_n_t = null_cache.get(bkey, bucket_vals[bkey])
            r2_perm = _max_r2_perm(Gk_t, null_n_t, perm_chunk_size)
            n_null_fits += 1
            if beta_approx:
                bs1, bs2, true_dof = cis.fit_beta_parameters(r2_perm, dof)       # core.py:358
            else:
                bs1 = bs2 = true_dof = np.nan

            # (b) min-p per B (cis.py:60-63)
            r2k_t = rk_t.pow(2)
            r2k_t = torch.where(torch.isnan(r2k_t),
                                torch.full_like(r2k_t, -1.0), r2k_t)             # cis.py:61
            ixc_t = r2k_t.argmax(0)                                              # cis.py:63
            r_nominal = rk_t.gather(0, ixc_t.unsqueeze(0)).squeeze(0).cpu().numpy()
            row = ixc_t.cpu().numpy()
            if keep is not None:
                row = keep[row]
            row_t = torch.as_tensor(row, device=device)
            cols_t = torch.as_tensor(cols, device=device)
            std_ratio = torch.sqrt(bvar_t[cols_t] / gvar_t[row_t]).cpu().numpy()  # cis.py:51

            # (c) prepare_cis_output's arithmetic (cis.py:566-599), vectorized over B
            r2_nominal = r_nominal * r_nominal                                   # cis.py:566
            slope = r_nominal * std_ratio                                        # cis.py:569
            tstat2 = dof * r2_nominal / (1 - r2_nominal)                         # cis.py:570
            slope_se = np.abs(slope) / np.sqrt(tstat2)                           # cis.py:571
            # exact vectorization of (np.sum(r2_perm >= r2_nominal)+1)/(nperm+1), cis.py:567
            n_ge = nperm - np.searchsorted(np.sort(r2_perm), r2_nominal, side='left')
            pval_perm = (n_ge + 1) / (nperm + 1)
            r2_64 = r2_nominal.astype(np.float64)
            pval_nominal = cis.pval_from_corr(r2_64, dof)                        # core.py:339
            if beta_approx:
                pval_true_df = cis.pval_from_corr(r2_64, true_dof)               # core.py:398
                pval_beta = stats.beta.cdf(pval_true_df, bs1, bs2)               # core.py:399
            else:
                pval_true_df = np.full(len(cols), np.nan)
                pval_beta = np.full(len(cols), np.nan)

            gvix = genotype_range[row]                                           # cis.py:720
            df = pd.DataFrame({
                'gene_a': gene_a,
                'gene_b': expr_df.index.values[cols],
                'num_var': num_var,
                'beta_shape1': bs1,
                'beta_shape2': bs2,
                'true_df': true_dof,
                'pval_true_df': pval_true_df,
                'variant_id': vid_all[gvix],
                'start_distance': vpos_all[gvix] - igc.phenotype_start[gene_a],  # cis.py:722
                'end_distance': vpos_all[gvix] - igc.phenotype_end[gene_a],      # cis.py:723
                'ma_samples': ma_samples_t[row_t].cpu().numpy(),
                'ma_count': ma_count_t[row_t].cpu().numpy(),
                'af': af_t[row_t].cpu().numpy(),
                'pval_nominal': pval_nominal,
                'slope': slope,
                'slope_se': slope_se,
                'pval_perm': pval_perm,
                'pval_beta': pval_beta,
                'same_chrom': b_chrom[cols] == a_chrom,
                'leakage_removed': n_var_full - num_var,
            })
            if run_eigenmt:
                df['tests_emt'] = m_eff
            out.append(df)

        del Gn_t, r_all_t

    if len(out) == 0:
        raise ValueError('No pairs produced results.')
    res_df = pd.concat(out, ignore_index=True)
    res_df.index = res_df['gene_a'] + PAIR_SEP + res_df['gene_b']
    res_df.index.name = 'pair_id'
    # keep cis.map_cis's column order so the two outputs line up column-for-column
    lead = ['gene_a', 'gene_b'] + CIS_OUTPUT_COLUMNS
    res_df = res_df[lead + [c for c in res_df.columns if c not in lead]]

    n_missing = n_pairs - len(res_df)
    if n_missing:
        n_drop_b = sum(int(b_const[b_ix_by_a[a]].sum()) for a in usable_a)
        other = n_missing - n_drop_a - n_drop_b
        logger.write(f'    ** {n_missing} of {n_pairs} pairs missing from output: '
                     f'{n_drop_a} gene A unusable, {n_drop_b} gene B constant, '
                     f'{other} other (no valid variants after filters / cis_window_b)')
    if annotate_leakage_window is not None:
        _annotate_leakage(res_df, variant_df, phenotype_pos_df, annotate_leakage_window)

    logger.write(f'  * {n_null_fits} permutation nulls fitted for {len(res_df)} edges '
                 f'({n_null_fits / max(len(usable_a), 1):.1f} per locus; '
                 f'the reference would need {len(res_df)})')
    logger.write(f'  Time elapsed: {(time.time() - start_time) / 60:.2f} min')
    logger.write('done.')
    # astype() raises KeyError on dict keys absent from the frame -> filter
    dtypes = {k: v for k, v in cis.output_dtype_dict.items() if k in res_df.columns}
    return res_df.astype(dtypes).infer_objects()


# ---------------------------------------------------------------------------------------
# across-edge multiple testing (step 3)
# ---------------------------------------------------------------------------------------

def bh_edges(res_df, fdr=0.05, pval_col='pval_beta'):
    """Benjamini-Hochberg across edges; adds 'qval_bh' and 'significant'.

    Uses ``eigenmt.padjust_bh`` (eigenmt.py:175), which needs no R. ``post.calculate_qvalues``
    (post.py:17) also accepts this frame as-is and logs the Beta-vs-empirical correlation as
    a calibration diagnostic, but requires R + qvalue + rpy2 and Storey's pi0 is unreliable
    on a few hundred tests.

    Note ``pval_perm`` is floored at 1/(nperm+1) (cis.py:567) and so cannot support
    edge-level FDR: BH over E edges needs p ~ 0.05/E, which is below that floor for any
    module of more than a few dozen genes. Always use the Beta-approximated p-value.
    """
    if pval_col == 'pval_perm':
        raise ValueError('pval_perm is floored at 1/(nperm+1) and cannot support edge-level '
                         'FDR; use pval_beta (run with beta_approx=True)')
    ok = res_df[pval_col].notnull()
    if not ok.all():
        print(f'    ** {int((~ok).sum())} edges with null {pval_col} excluded from BH')
    out_df = res_df.copy()
    out_df['qval_bh'] = np.nan
    out_df.loc[ok, 'qval_bh'] = eigenmt.padjust_bh(res_df.loc[ok, pval_col].values)
    out_df['significant'] = out_df['qval_bh'] <= fdr
    print(f'  * edges @ FDR {fdr}: {int(out_df["significant"].sum())} / {int(ok.sum())}')
    return out_df
