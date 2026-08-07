# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

tensorQTL is a GPU-accelerated *cis*- and *trans*-QTL mapper (PyTorch backend). It is a library **and** a CLI; both entry points call the same `map_*` functions. Version is read from installed package metadata (`importlib.metadata.version("tensorqtl")`), so the package must be pip-installed (`pip install -e .`) for `__init__.py` to import.

## Commands

There is no test suite, linter, or CI in this repo. Verification is done by running the examples in `example/tensorqtl_examples.ipynb` against the bundled chr18 GEUVADIS data in `example/data/`:

```bash
cd example
python3 -m tensorqtl \
    data/GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18 \
    data/GEUVADIS.445_samples.expression.bed.gz \
    GEUVADIS.445_samples \
    --covariates data/GEUVADIS.445_samples.covariates.txt \
    --mode cis
```

Modes: `cis`, `cis_nominal`, `cis_independent`, `cis_susie`, `trans`, `trans_susie` (see `python3 -m tensorqtl --help` and README for the per-mode required flags — e.g. `cis_independent` and `cis_susie` require `--cis_output`).

Environment: `mamba env create -f install/tensorqtl_env.yml`. GPU setup notes in `install/INSTALL.md`; CUDA image in `Dockerfile`.

## Architecture

Layering (each module imports the one above it):

- `core.py` — device-agnostic primitives shared by every mapper: `Residualizer` (QR-based covariate projection; carries `dof`), `calculate_corr`, `get_allele_stats`, MAF filters, `impute_mean`, the Beta-approximation of empirical p-values (`fit_beta_parameters`, `calculate_beta_approx_pval`), `read_phenotype_bed`, and `SimpleLogger`. Also sets the module-global `has_rpy2` flag at import time by shelling out to `R -e 'library(qvalue)'`.
- `pgen.py` / `genotypeio.py` — genotype I/O. `genotypeio.load_genotypes()` is the format dispatcher (pgen/psam/pvar → `pgen.PgenReader`; bed/bim/fam → `PlinkReader` via `pandas_plink`; parquet/BED/tsv.gz → DataFrame with `variant_df = None`, trans-only). `InputGeneratorCis` is the workhorse for cis modes: it computes per-phenotype cis-window variant index ranges once, drops phenotypes that are constant / on contigs without genotypes / without variants in-window, then yields `(phenotype, genotypes, genotype_range, phenotype_id)` from a prefetching background thread (`@background(max_prefetch=6)`).
- `cis.py`, `trans.py`, `susie.py`, `coloc.py`, `mixqtl.py` — the mappers. `eigenmt.py` (M<sub>eff</sub> estimation) and `post.py` (`calculate_qvalues`, `get_significant_pairs`, `calculate_replication`, aFC) are called from them.
- `tensorqtl.py` — CLI `main()`; parses args, loads inputs once, dispatches on `--mode`, writes outputs. `__main__.py` just calls it.

Data flow in every mapper is the same shape: pandas DataFrames in (genotypes × samples, phenotypes × samples, samples × covariates) → per-phenotype or per-batch tensors moved to `device` → correlation/regression on GPU → results accumulated back into DataFrames.

### Conventions to follow

- **Flat, non-relative imports.** Every module does `sys.path.insert(1, os.path.dirname(__file__))` then `from core import *` / `import genotypeio, cis`. Do not "fix" these to relative imports — `tensorqtl.py` and the CLI depend on the flat namespace.
- **`_t` suffix = torch tensor**, everything else is numpy/pandas. `_df`/`_s` = DataFrame/Series.
- **Orientations are load-bearing and asserted.** Genotypes and phenotypes are *rows × samples*; covariates and interactions are *samples × columns*. `map_*` functions assert that sample orders match (`phenotype_df.columns` vs `covariates_df.index`) — keep those assertions when editing.
- **Device selection is per-function**, not global: each `map_*` re-evaluates `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Degrees of freedom** come from `Residualizer.dof` (already accounts for intercept + genotype term); interaction models subtract `2*ni` on top of that. Off-by-one here silently corrupts p-values.
- **Output dtypes** for cis results are centralized in `core.output_dtype_dict`; column descriptions live in `docs/outputs.md` — update it when adding output columns.

### Cross-cutting behaviors

- **Optional R.** q-values (`post.calculate_qvalues`), `--logp` (-log10 p-values), and `rfunc.py` all require R + `qvalue` + rpy2. Code paths must degrade gracefully when `has_rpy2` is False, as the CLI does before calling `calculate_qvalues`.
- **Chunked mode.** `--chunk_size` requires pgen input and switches from "load all genotypes" to streaming via `genotypeio.generate_paired_chunks(pgr, ...)`. Each mode implements *both* branches in `tensorqtl.py`, and `cis_nominal` additionally renames/concatenates the per-chunk parquet files afterward. New cis modes need the same pair of branches.
- **Output files.** Per-chromosome parquet for nominal pairs (`<prefix>.cis_qtl_pairs.<chr>.parquet`), gzipped TSV for phenotype-level summaries, pickle + parquet for SuSiE. Interaction mode's `write_top`/`write_stats` flags control which of the two nominal outputs are produced (`--best_only` on the CLI disables full stats).

## Upstream

This is a fork of `broadinstitute/tensorqtl`; commits here are upstream's. Prefer changes that stay mergeable with upstream (`git log --oneline` shows the shared history).
