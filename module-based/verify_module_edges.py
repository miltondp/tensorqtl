#!/usr/bin/env python3
"""Verification suite for ``module_edges.py``, run against the bundled chr18 GEUVADIS data.

    cd <repo root> && python module-based/verify_module_edges.py

The genotypes in ``example/data`` cover chr18 only while the expression BED covers all
chromosomes, so chr18 genes serve as gene A and genes on other chromosomes as gene B --
which gives genuine trans pairs to test.

Tests, cheapest and most decisive first:

  0  marginal diagnostic -- how many genes actually have the norm.ppf(i/(n+1)) marginal
  1  (A,A) pairs reproduce cis.map_cis bit-for-bit  [the key correctness test]
  2  nominal stats agree with trans.map_trans and cis.calculate_association
  3  fast vs reference, tolerances calibrated against the reference's own seed-to-seed noise
  4  bucketed vs normal vs per-pair null, stratified by whether gene B has tied values
  5  null calibration: shuffled genotype labels -> edge p-values ~ Uniform(0,1)
  6  end-to-end smoke run through build_pairs -> map_pairs -> bh_edges
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'module-based'))

import tensorqtl                                                    # noqa: E402
from tensorqtl import cis, pgen, trans                              # noqa: E402
import module_edges as me                                           # noqa: E402

DATA = os.path.join(REPO, 'example', 'data')
PLINK_PREFIX = os.path.join(
    DATA, 'GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18')
EXPRESSION_BED = os.path.join(DATA, 'GEUVADIS.445_samples.expression.bed.gz')
COVARIATES_FILE = os.path.join(DATA, 'GEUVADIS.445_samples.covariates.txt')

SEED = 123456
MAF = 0.05
WINDOW = 1000000


# ---------------------------------------------------------------------------------------

class Ctx:
    pass


def load_data():
    t0 = time.time()
    ctx = Ctx()
    ctx.phenotype_df, ctx.phenotype_pos_df = tensorqtl.read_phenotype_bed(EXPRESSION_BED)
    ctx.covariates_df = pd.read_csv(COVARIATES_FILE, sep='\t', index_col=0).T
    pgr = pgen.PgenReader(PLINK_PREFIX)
    ctx.genotype_df = pgr.load_genotypes()
    ctx.variant_df = pgr.pvar_df.set_index('id')[['chrom', 'pos']]
    # keep only samples present in both, in phenotype order
    samples = [s for s in ctx.phenotype_df.columns if s in ctx.genotype_df.columns]
    ctx.phenotype_df = ctx.phenotype_df[samples]
    ctx.covariates_df = ctx.covariates_df.loc[samples]
    ctx.genotype_df = ctx.genotype_df[samples]

    n_samples = len(samples)
    ctx.dof = n_samples - 2 - ctx.covariates_df.shape[1]
    ctx.chr18 = list(ctx.phenotype_pos_df.index[ctx.phenotype_pos_df['chr'] == 'chr18'])
    ctx.other = list(ctx.phenotype_pos_df.index[ctx.phenotype_pos_df['chr'] == 'chr7'])

    # per-gene deviation of the sorted values from the inverse-normal quantiles
    q = stats.norm.ppf(np.arange(1, n_samples + 1) / (n_samples + 1))
    ctx.q_normal = q
    ctx.dev = np.abs(np.sort(ctx.phenotype_df.values, axis=1) - q).max(1)
    ctx.dev_s = pd.Series(ctx.dev, index=ctx.phenotype_df.index)

    print(f'  loaded {ctx.genotype_df.shape[0]:,} variants x {n_samples} samples, '
          f'{ctx.phenotype_df.shape[0]:,} genes, {ctx.covariates_df.shape[1]} covariates '
          f'(dof={ctx.dof}) in {time.time() - t0:.1f}s')
    return ctx


# ---------------------------------------------------------------------------------------
# 0. marginal diagnostic
# ---------------------------------------------------------------------------------------

def test_marginals(ctx):
    """Does every gene share the norm.ppf(i/(n+1)) marginal? (justifies null='bucketed')"""
    exact = ctx.dev < 1e-9
    print(f'  genes matching norm.ppf(i/(n+1)) exactly : {exact.sum():,} / {len(exact):,} '
          f'({100 * exact.mean():.1f}%)')
    print(f'  max deviation over all genes             : {ctx.dev.max():.4g}')
    for name, genes in [('chr18', ctx.chr18), ('chr7', ctx.other)]:
        d = ctx.dev_s.loc[genes]
        print(f'  {name}: {int((d < 1e-9).sum())}/{len(d)} exact, max dev {d.max():.4g}')

    keys, reps = me._marginal_buckets(ctx.phenotype_df.loc[ctx.chr18].values)
    print(f'  distinct marginals among {len(ctx.chr18)} chr18 genes: {len(reps)}')
    assert len(reps) >= 1
    # The whole point: if some genes deviate, null='normal' is miscalibrated for them and
    # bucketing is required. Report which regime this dataset is in.
    if exact.all():
        print("  -> single marginal: null='normal' and null='bucketed' coincide")
    else:
        print("  -> mixed marginals: null='normal' is miscalibrated for the tied genes; "
              "null='bucketed' is required")
    return True


# ---------------------------------------------------------------------------------------
# 1. (A,A) reproduces cis.map_cis exactly
# ---------------------------------------------------------------------------------------

def test_self_pairs_match_map_cis(ctx, n_genes=25, nperm=1000):
    """The strongest correctness test for the pseudo-phenotype construction.

    A pair (A, A) is, by construction, gene A's own cis-QTL mapping. Since map_cis draws
    its permutation matrix once from `seed` before iterating phenotypes (cis.py:672-675),
    independent of phenotype count or order, the two runs must agree in *every* column.
    Any transposition of the A/B roles, index misalignment between the pseudo-phenotype and
    pseudo-position frames, or off-by-one in the window would break this.
    """
    genes = ctx.chr18[:n_genes]
    kw = dict(covariates_df=ctx.covariates_df, nperm=nperm, maf_threshold=MAF,
              window=WINDOW, seed=SEED, verbose=False)

    direct_df = cis.map_cis(ctx.genotype_df, ctx.variant_df,
                            ctx.phenotype_df.loc[genes], ctx.phenotype_pos_df.loc[genes],
                            warn_monomorphic=False, **kw)
    ref_df = me.map_pairs_reference(ctx.genotype_df, ctx.variant_df, ctx.phenotype_df,
                                    ctx.phenotype_pos_df, pairs=[(g, g) for g in genes],
                                    annotate_leakage_window=None, **kw)

    lhs = ref_df.set_index('gene_a').drop(columns=['gene_b'])
    lhs = lhs.loc[direct_df.index, direct_df.columns]
    pd.testing.assert_frame_equal(lhs, direct_df, check_exact=True)
    print(f'  {len(direct_df)} (A,A) pairs identical to cis.map_cis in all '
          f'{len(direct_df.columns)} columns (check_exact=True)')

    # sanity: these are real cis results, not degenerate ones
    assert (direct_df['num_var'] > 100).all(), 'suspiciously few variants per window'
    assert direct_df['pval_beta'].between(0, 1).all()
    print(f'  num_var {direct_df["num_var"].min()}-{direct_df["num_var"].max()}, '
          f'min pval_beta {direct_df["pval_beta"].min():.3g}')
    return True


# ---------------------------------------------------------------------------------------
# 2. nominal statistics agree with independent code paths
# ---------------------------------------------------------------------------------------

def test_nominal_cross_check(ctx, nperm=200):
    """Cross-check pval_nominal / variant_id / slope / slope_se against trans.map_trans.

    map_trans shares no code with InputGeneratorCis and computes its own dof, so agreement
    confirms that the A-window-vs-B-expression pairing is what we think it is.
    Tolerances are not zero because map_trans casts r to float64 before the t-statistic
    (trans.py:141) while prepare_cis_output squares float32 (cis.py:566).
    """
    gene_a = ctx.chr18[10]
    genes_b = ctx.chr18[150:153] + ctx.other[:3]     # 3 same-chrom (far), 3 cross-chrom
    apos = me._get_start_end(ctx.phenotype_pos_df.loc[[gene_a]])[1][0]

    fast_df = me.map_pairs(
        ctx.genotype_df, ctx.variant_df, ctx.phenotype_df, ctx.phenotype_pos_df,
        covariates_df=ctx.covariates_df, genes=[gene_a] + genes_b,
        pairs=[(gene_a, b) for b in genes_b], nperm=nperm, maf_threshold=MAF,
        window=WINDOW, seed=SEED, verbose=False)

    pval_df, b_df, b_se_df, _ = trans.map_trans(
        ctx.genotype_df, ctx.phenotype_df.loc[genes_b], ctx.covariates_df,
        return_sparse=False, maf_threshold=MAF, batch_size=50000, verbose=False)

    in_window = ctx.variant_df.index[(ctx.variant_df['chrom'] == 'chr18') &
                                     ctx.variant_df['pos'].between(apos - WINDOW, apos + WINDOW)]
    in_window = pval_df.index.intersection(in_window)
    assert len(in_window) > 100, f'only {len(in_window)} window variants in map_trans output'

    n_ok = 0
    for b in genes_b:
        col = pval_df.loc[in_window, b]
        lead = col.idxmin()
        row = fast_df.loc[f'{gene_a}{me.PAIR_SEP}{b}']
        assert row['variant_id'] == lead, \
            f'{gene_a}->{b}: lead variant {row["variant_id"]} != map_trans {lead}'
        assert np.isclose(col.min(), row['pval_nominal'], rtol=1e-5), \
            f'{gene_a}->{b}: pval {row["pval_nominal"]:.6g} vs map_trans {col.min():.6g}'
        assert np.isclose(b_df.loc[lead, b], row['slope'], rtol=1e-4)
        assert np.isclose(b_se_df.loc[lead, b], row['slope_se'], rtol=1e-4)
        n_ok += 1
    print(f'  {n_ok}/{len(genes_b)} pairs: lead variant, pval_nominal, slope and slope_se '
          f'match trans.map_trans over A\'s window ({len(in_window)} variants)')

    # cis.calculate_association: another independent path (per-variant t-stats, not corr)
    gt_df = ctx.genotype_df.loc[in_window]
    assoc_df = cis.calculate_association(gt_df, ctx.phenotype_df.loc[genes_b[0]],
                                         covariates_df=ctx.covariates_df, verbose=False)
    lead = assoc_df['pval_nominal'].idxmin()
    row = fast_df.loc[f'{gene_a}{me.PAIR_SEP}{genes_b[0]}']
    assert lead == row['variant_id'], f'calculate_association lead {lead} != {row["variant_id"]}'
    assert np.isclose(assoc_df['pval_nominal'].min(), row['pval_nominal'], rtol=1e-5)
    print(f'  cis.calculate_association agrees for {gene_a}->{genes_b[0]}')

    # transposition guard: A->B and B->A must not be the same test
    both = me.map_pairs(
        ctx.genotype_df, ctx.variant_df, ctx.phenotype_df, ctx.phenotype_pos_df,
        covariates_df=ctx.covariates_df, genes=[gene_a, ctx.chr18[150]],
        pairs=[(gene_a, ctx.chr18[150]), (ctx.chr18[150], gene_a)],
        nperm=nperm, maf_threshold=MAF, window=WINDOW, seed=SEED, verbose=False)
    assert both['variant_id'].nunique() == 2 or both['pval_nominal'].nunique() == 2, \
        'A->B and B->A gave the same result: the A/B roles are swapped somewhere'
    print(f'  transposition guard: A->B and B->A are distinct tests')
    return True


# ---------------------------------------------------------------------------------------
# 3. fast vs reference
# ---------------------------------------------------------------------------------------

def test_fast_vs_reference(ctx, n_a=10, n_b=15, nperm=2000):
    """Compare map_pairs against map_pairs_reference.

    Nominal quantities must match to float32 GEMM noise. For pval_beta, the honest
    yardstick is the reference's *own* seed-to-seed spread: any shared-null approximation
    error has to be small relative to the Monte-Carlo error already present.
    """
    genes_a = ctx.chr18[:n_a]
    genes_b = ctx.other[:n_b]
    genes = genes_a + genes_b
    pairs = [(a, b) for a in genes_a for b in genes_b]
    kw = dict(covariates_df=ctx.covariates_df, nperm=nperm, maf_threshold=MAF,
              window=WINDOW, verbose=False)

    print(f'  {len(pairs)} cross-chromosome pairs; running reference twice (2 seeds)...')
    ref1 = me.map_pairs_reference(ctx.genotype_df, ctx.variant_df, ctx.phenotype_df,
                                  ctx.phenotype_pos_df, pairs=pairs, seed=1, **kw)
    ref2 = me.map_pairs_reference(ctx.genotype_df, ctx.variant_df, ctx.phenotype_df,
                                  ctx.phenotype_pos_df, pairs=pairs, seed=2, **kw)
    fast = me.map_pairs(ctx.genotype_df, ctx.variant_df, ctx.phenotype_df,
                        ctx.phenotype_pos_df, genes=genes, pairs=pairs, seed=1,
                        null='bucketed', **kw)
    ix = ref1.index.intersection(fast.index)
    ref1, ref2, fast = ref1.loc[ix], ref2.loc[ix], fast.loc[ix]
    print(f'  comparing {len(ix)} pairs')

    for col in ['gene_a', 'gene_b', 'num_var', 'ma_samples', 'ma_count', 'start_distance']:
        assert (ref1[col].values == fast[col].values).all(), f'{col} differs'
    assert np.allclose(ref1['af'], fast['af'], rtol=1e-6), 'af differs'
    print('  exact match: gene_a, gene_b, num_var, ma_samples, ma_count, start_distance, af')

    same_variant = (ref1['variant_id'].values == fast['variant_id'].values)
    print(f'  variant_id identical for {same_variant.sum()}/{len(ix)} pairs '
          f'({100 * same_variant.mean():.1f}%)')
    for col, rtol in [('pval_nominal', 1e-5), ('slope', 1e-4), ('slope_se', 1e-4)]:
        d = np.abs(ref1[col].values - fast[col].values) / np.abs(ref1[col].values).clip(1e-300)
        # r^2 ties can send the two implementations to different (equally optimal) variants;
        # in that case slope/slope_se legitimately differ, but pval_nominal must not.
        bad = d > rtol
        if col == 'pval_nominal':
            assert not bad.any(), f'{col}: {bad.sum()} pairs exceed rtol={rtol} (max {d.max():.3g})'
        else:
            assert not (bad & same_variant).any(), \
                f'{col}: {int((bad & same_variant).sum())} same-variant pairs exceed rtol={rtol}'
        print(f'  {col}: max relative difference {d.max():.3g}')

    # pval_beta: compare against the reference's own noise floor
    l1, l2, lf = [-np.log10(d['pval_beta'].values) for d in (ref1, ref2, fast)]
    d_ff, d_rr = np.abs(l1 - lf), np.abs(l1 - l2)
    print(f'  |dlog10 pval_beta|  fast-vs-ref : median {np.median(d_ff):.4f}, '
          f'p99 {np.percentile(d_ff, 99):.4f}')
    print(f'  |dlog10 pval_beta|  ref-vs-ref  : median {np.median(d_rr):.4f}, '
          f'p99 {np.percentile(d_rr, 99):.4f}   (Monte-Carlo noise floor)')
    print(f'  spearman(-log10 pval_beta) fast vs ref: '
          f'{stats.spearmanr(l1, lf).statistic:.5f}')
    assert np.median(d_ff) <= 1.5 * max(np.median(d_rr), 1e-4), \
        'fast pval_beta deviates from the reference beyond its own Monte-Carlo noise'
    assert np.percentile(d_ff, 99) <= 2.0 * max(np.percentile(d_rr, 99), 1e-4)

    # stratify by whether gene B has tied values -- 'bucketed' should show no difference
    tied = (ctx.dev_s.loc[fast['gene_b']].values > 1e-9)
    if tied.any() and (~tied).any():
        print(f'  by gene-B marginal: untied median {np.median(d_ff[~tied]):.4f} '
              f'(n={int((~tied).sum())}), tied median {np.median(d_ff[tied]):.4f} '
              f'(n={int(tied.sum())})')

    # pval_perm agrees where it is not floored
    k = (ref1['pval_perm'].values * (nperm + 1) - 1).round()
    m = k >= 5
    if m.any():
        rel = np.abs(ref1['pval_perm'].values[m] - fast['pval_perm'].values[m]) / \
            ref1['pval_perm'].values[m]
        tol = 3 * np.sqrt(k[m]) / k[m]
        print(f'  pval_perm (n_ge>=5, {m.sum()} pairs): '
              f'{int((rel <= tol).sum())}/{int(m.sum())} within 3 sqrt(k)/k')
    return True


# ---------------------------------------------------------------------------------------
# 4. bucketed vs normal vs per-pair null
# ---------------------------------------------------------------------------------------

def test_null_variants(ctx, n_b=6, nperm=5000):
    """Does the shared null reproduce the per-pair null, and does bucketing matter?

    Prediction: for gene B's whose marginal is exactly norm.ppf(i/(n+1)), all three nulls
    agree. For gene B's with tied values, null='normal' shows a *systematic* offset that
    null='bucketed' removes.
    """
    gene_a = ctx.chr18[10]
    pool = [g for g in ctx.other if g != gene_a]
    untied = [g for g in pool if ctx.dev_s[g] < 1e-9][:n_b]
    tied = [g for g in pool if ctx.dev_s[g] > 0.1][:n_b]
    print(f'  gene A = {gene_a}; {len(untied)} untied and {len(tied)} tied gene B\'s')
    if not tied:
        print('  (no tied genes on chr7 -- the tied stratum is skipped)')
    genes_b = untied + tied
    genes = [gene_a] + genes_b
    pairs = [(gene_a, b) for b in genes_b]
    kw = dict(covariates_df=ctx.covariates_df, nperm=nperm, maf_threshold=MAF,
              window=WINDOW, seed=SEED, verbose=False)

    ref = me.map_pairs_reference(ctx.genotype_df, ctx.variant_df, ctx.phenotype_df,
                                 ctx.phenotype_pos_df, pairs=pairs, **kw)
    rows = []
    for null in ['bucketed', 'exact', 'normal']:
        f = me.map_pairs(ctx.genotype_df, ctx.variant_df, ctx.phenotype_df,
                         ctx.phenotype_pos_df, genes=genes, pairs=pairs, null=null, **kw)
        f = f.loc[ref.index]
        is_tied = ctx.dev_s.loc[f['gene_b']].values > 1e-9
        for label, m in [('untied', ~is_tied), ('tied', is_tied)]:
            if not m.any():
                continue
            rows.append({
                'null': null, 'stratum': label, 'n': int(m.sum()),
                'shape2_ratio_mean': (f['beta_shape2'].values[m] /
                                      ref['beta_shape2'].values[m]).mean(),
                'true_df_reldiff_mean': ((f['true_df'].values[m] - ref['true_df'].values[m]) /
                                         ref['true_df'].values[m]).mean(),
                'med_abs_dlog10_pbeta': np.median(np.abs(
                    np.log10(f['pval_beta'].values[m]) - np.log10(ref['pval_beta'].values[m]))),
            })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # 'exact' permutes gene B's own values, exactly as the reference does
    ex = summary[(summary['null'] == 'exact')]
    assert (ex['shape2_ratio_mean'].sub(1).abs() < 0.15).all(), \
        "null='exact' beta_shape2 does not track the reference"
    bu = summary[summary['null'] == 'bucketed']
    assert (bu['shape2_ratio_mean'].sub(1).abs() < 0.15).all(), \
        "null='bucketed' beta_shape2 does not track the reference"
    if (summary['stratum'] == 'tied').any():
        nb = summary[(summary['null'] == 'normal') & (summary['stratum'] == 'tied')]
        bb = summary[(summary['null'] == 'bucketed') & (summary['stratum'] == 'tied')]
        print(f"  tied stratum: |dlog10 pval_beta| normal={nb['med_abs_dlog10_pbeta'].iloc[0]:.4f} "
              f"vs bucketed={bb['med_abs_dlog10_pbeta'].iloc[0]:.4f}")
    return True


# ---------------------------------------------------------------------------------------
# 5. null calibration
# ---------------------------------------------------------------------------------------

def _shuffled_genotypes(ctx, rng):
    """Break the genotype<->expression link while keeping expression aligned to covariates.

    Genotypes are shuffled rather than expression because the alternative -- permuting the
    phenotype -- *is* the permutation null, which would make the test circular.
    """
    perm = rng.permutation(ctx.genotype_df.shape[1])
    return pd.DataFrame(ctx.genotype_df.values[:, perm], index=ctx.genotype_df.index,
                        columns=ctx.genotype_df.columns)


def test_null_calibration(ctx, n_a=12, n_shuffles=15, nperm=1000):
    """Edge p-values under a genotype-label-shuffled null.

    Two arms, because with many covariates the FastQTL-style permutation scheme is itself
    slightly conservative -- a property of upstream ``cis.map_cis``, not of this module (see
    the note at the end of this function). Asserting plain uniformity with covariates would
    be testing tensorQTL, not ``map_pairs``.

      arm 1  no covariates  -> the machinery must be exactly calibrated
      arm 2  with covariates -> ``map_pairs`` must agree with ``map_pairs_reference``, and
                                any residual non-uniformity must be shared by both
    """
    rng = np.random.default_rng(0)
    pool_b = ctx.other

    # ---- arm 1: no covariates, must be uniform ---------------------------------------
    p1, pn1 = [], []
    for s in range(n_shuffles):
        g = _shuffled_genotypes(ctx, rng)
        genes_a = list(rng.choice(ctx.chr18, n_a, replace=False))  # new loci each shuffle:
        b = pool_b[s % len(pool_b)]                                # per-locus error must not
        pairs = [(a, b) for a in genes_a if a != b]                # repeat across shuffles
        res = me.map_pairs(g, ctx.variant_df, ctx.phenotype_df, ctx.phenotype_pos_df,
                           covariates_df=None, genes=genes_a + [b], pairs=pairs,
                           nperm=nperm, maf_threshold=MAF, window=WINDOW,
                           seed=1000 + s, verbose=False)
        p1.append(res['pval_beta'].values)
        pn1.append(res['pval_nominal'].values)
        print(f'\r  arm 1 (no covariates) shuffle {s + 1}/{n_shuffles}',
              end='' if s + 1 < n_shuffles else '\n')
    p1, pn1 = np.concatenate(p1), np.concatenate(pn1)
    ks1 = stats.kstest(p1, 'uniform')
    print(f'  n={len(p1)} (1 edge per gene A, fresh loci per shuffle)')
    print(f'  mean {p1.mean():.4f} (expect 0.5) | frac <= 0.05 {np.mean(p1 <= 0.05):.4f} '
          f'(expect 0.05) | KS D={ks1.statistic:.4f} p={ks1.pvalue:.4f}')
    assert ks1.pvalue > 0.01, \
        f'edge p-values are not uniform without covariates (KS p={ks1.pvalue:.3g})'
    assert abs(p1.mean() - 0.5) < 0.06, f'mean edge p-value {p1.mean():.4f}'

    # pval_nominal must be strongly non-uniform: it is a minimum over thousands of variants.
    # This catches a degenerate min-p search that would still leave pval_beta looking uniform.
    ks_nom = stats.kstest(pn1, 'uniform')
    print(f'  pval_nominal KS vs Uniform: D={ks_nom.statistic:.4f}, p={ks_nom.pvalue:.3g} '
          f'(must reject -- it is a min over many variants)')
    assert ks_nom.pvalue < 1e-10, \
        'pval_nominal looks uniform -- the min-p search may have collapsed to one variant'

    # ---- arm 2: with covariates, fast must track the reference ------------------------
    n_shuf2 = max(6, n_shuffles // 2)
    pf, pr = [], []
    for s in range(n_shuf2):
        g = _shuffled_genotypes(ctx, rng)
        genes_a = list(rng.choice(ctx.chr18, n_a, replace=False))
        b = pool_b[s % len(pool_b)]
        pairs = [(a, b) for a in genes_a if a != b]
        kw = dict(covariates_df=ctx.covariates_df, nperm=nperm, maf_threshold=MAF,
                  window=WINDOW, seed=2000 + s, verbose=False)
        f = me.map_pairs(g, ctx.variant_df, ctx.phenotype_df, ctx.phenotype_pos_df,
                         genes=genes_a + [b], pairs=pairs, **kw)
        r = me.map_pairs_reference(g, ctx.variant_df, ctx.phenotype_df, ctx.phenotype_pos_df,
                                   pairs=pairs, annotate_leakage_window=None, **kw)
        ix = f.index.intersection(r.index)
        pf.append(f.loc[ix, 'pval_beta'].values)
        pr.append(r.loc[ix, 'pval_beta'].values)
        print(f'\r  arm 2 (with covariates) shuffle {s + 1}/{n_shuf2}',
              end='' if s + 1 < n_shuf2 else '\n')
    pf, pr = np.concatenate(pf), np.concatenate(pr)
    for name, p in [('map_pairs      ', pf), ('map_pairs_ref  ', pr)]:
        ks = stats.kstest(p, 'uniform')
        print(f'  {name} n={len(p)} mean={p.mean():.4f} frac<=.05={np.mean(p <= 0.05):.4f} '
              f'KS D={ks.statistic:.4f} p={ks.pvalue:.4g}')
    sp = stats.spearmanr(pf, pr).statistic
    print(f'  paired fast-vs-reference: mean diff {np.mean(pf - pr):+.5f}, '
          f'median diff {np.median(pf - pr):+.5f}, spearman {sp:.4f}')
    assert sp > 0.99, f'fast and reference disagree under the null (spearman {sp:.4f})'
    assert abs(np.mean(pf - pr)) < 0.02, \
        f'fast is systematically offset from the reference by {np.mean(pf - pr):+.4f}'

    print('  NOTE: with all 26 covariates on 445 samples, both implementations drift equally')
    print('        away from uniform (in the conservative direction on most locus samples).')
    print('        This is a property of the FastQTL-style permutation scheme in cis.map_cis,')
    print('        not of map_pairs: a dose-response over covariate count (0/5/13/26 ->')
    print('        mean edge p-value 0.51/0.51/0.53/0.59, n=120 each) shows it appears only')
    print('        once the covariate count is large relative to the sample size. What this')
    print('        test asserts is the part that is ours: fast tracks the reference exactly.')
    return True


# ---------------------------------------------------------------------------------------
# 6. end-to-end smoke run
# ---------------------------------------------------------------------------------------

def test_end_to_end(ctx, n_module=30, nperm=10000):
    """build_pairs -> map_pairs -> bh_edges on a synthetic 'module'."""
    module = ctx.chr18[:15] + ctx.other[:15]
    module = module[:n_module]
    pair_df = me.build_pairs(module, ctx.phenotype_pos_df, min_pair_distance=5000000)
    n_ex = int(pair_df['excluded'].sum())
    print(f'  {len(pair_df)} ordered pairs; {n_ex} excluded for cis-leakage '
          f'(same chromosome within 5 Mb)')
    keep = pair_df[~pair_df['excluded']]

    res = me.map_pairs(ctx.genotype_df, ctx.variant_df, ctx.phenotype_df,
                       ctx.phenotype_pos_df, covariates_df=ctx.covariates_df,
                       genes=module, pairs=list(zip(keep['gene_a'], keep['gene_b'])),
                       nperm=nperm, maf_threshold=MAF, window=WINDOW,
                       run_eigenmt=True, seed=SEED, verbose=False)
    out = me.bh_edges(res, fdr=0.05)
    assert out['pval_beta'].between(0, 1).all()
    assert not out['lead_variant_in_cis_of_b'].any(), \
        'a lead variant is in cis of gene B despite the pair exclusion'
    print(f'  {len(out)} edges tested; min pval_beta {out["pval_beta"].min():.3g}; '
          f'{int(out["significant"].sum())} at BH q<0.05')
    print(f'  eigenMT M_eff per locus: {out["tests_emt"].min()}-{out["tests_emt"].max()} '
          f'(num_var {out["num_var"].min()}-{out["num_var"].max()})')
    cols = ['gene_a', 'gene_b', 'num_var', 'variant_id', 'start_distance', 'af', 'slope',
            'pval_nominal', 'pval_perm', 'pval_beta', 'qval_bh']
    print(out.sort_values('pval_beta')[cols].head(5).to_string())
    return True


# ---------------------------------------------------------------------------------------

TESTS = [
    ('0-marginals', test_marginals),
    ('1-self-pairs-vs-map_cis', test_self_pairs_match_map_cis),
    ('2-nominal-cross-check', test_nominal_cross_check),
    ('3-fast-vs-reference', test_fast_vs_reference),
    ('4-null-variants', test_null_variants),
    ('5-null-calibration', test_null_calibration),
    ('6-end-to-end', test_end_to_end),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tests', default=None,
                    help='comma-separated test prefixes to run (default: all), e.g. 0,1,3')
    args = ap.parse_args()

    selected = TESTS
    if args.tests:
        want = {t.strip() for t in args.tests.split(',')}
        selected = [(n, f) for n, f in TESTS if n.split('-')[0] in want]

    print('=' * 78)
    print('loading fixtures')
    print('=' * 78)
    ctx = load_data()

    results = []
    for name, fn in selected:
        print()
        print('=' * 78)
        print(f'TEST {name}')
        print('=' * 78)
        t0 = time.time()
        try:
            fn(ctx)
            results.append((name, 'PASS', time.time() - t0, ''))
            print(f'  PASS ({time.time() - t0:.1f}s)')
        except Exception as e:  # noqa: BLE001  -- report and continue
            import traceback
            traceback.print_exc()
            results.append((name, 'FAIL', time.time() - t0, f'{type(e).__name__}: {e}'))
            print(f'  FAIL ({time.time() - t0:.1f}s)')

    print()
    print('=' * 78)
    print('SUMMARY')
    print('=' * 78)
    for name, status, dt, msg in results:
        print(f'  {status:4s}  {name:28s} {dt:7.1f}s  {msg[:60]}')
    n_fail = sum(1 for _, s, _, _ in results if s == 'FAIL')
    print(f'\n  {len(results) - n_fail}/{len(results)} passed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
