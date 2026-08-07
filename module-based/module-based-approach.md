> I have a gene module (i.e., gene co-expression network) obtained from gene expression data. I know this module is enriched with GWAS/TWAS genes for some relevant traits (meaning that some GWAS SNPs act as cis-eQTLs).  Now I want to infer a directed network from this module by comparing all genes in the module with the rest of the genes in the module like this: for all gene pairs A,B (i.e., gene A and gene B) in the module, I want to run tensorQTL (SNP-gene expression mapping) for all SNPs in gene A (1 mbps window) against the expression of gene B. With this, I aim to build a directed network, where there is an edge between two genes if there is at least one significant eQTL (which could be a trans-eQTL since genes in the module could be very far away each other, even in different chromosomes). My question is: what is the statistical procedure to build this network? should I consider multiple testing, if so, how? should I consider gene length? what else?

What you’re proposing is basically a **targeted trans-eQTL scan** where each “regulatory unit” is *the cis window around gene A*, and each “phenotype” is *expression of gene B*. The directed edge (A \rightarrow B) is then “there exists at least one SNP in A’s cis window associated with expression of B”.

That can be made statistically principled, but you *really* want to be explicit about **(i) what your null hypothesis is per edge** and **(ii) how you control error rates across the huge number of tested SNP–gene pairs and gene–gene edges**.

Below is a procedure that’s commonly used in QTL pipelines (cis and trans), adapted to your “A-locus → B-expression” edge definition.

---

## 1) Define the unit of inference: edge-level hypotheses (not SNP-pair-level)

For each ordered pair ((A,B)), define the null:

[
H_{0}^{A\rightarrow B}: \text{No variant in the cis window of gene A affects expression of gene B}
]

This is a **set-based null** (a locus has no effect on a phenotype), not a single SNP test. That matters because otherwise you’ll drown in multiple testing and gene-length bias.

So for each ((A,B)), you want **one calibrated p-value** for the *best* association in A’s window with B.

---

## 2) Within-edge multiple testing: correct across SNPs in A’s cis window (LD-aware)

Let (S_A) be all SNPs within (say) ±1 Mb of A’s TSS (your choice), after QC (MAF, INFO, HWE, missingness).

You will compute nominal p-values (p_{s,B}) for (s\in S_A), then take the minimum:
[
p_{\min} = \min_{s\in S_A} p_{s,B}
]

But (p_{\min}) is *not* uniform under the null because you searched over many SNPs (and they’re correlated by LD). You need an LD-aware adjustment to turn this into an **edge-level p-value** (p^{A\rightarrow B}_{\text{edge}}).

Two standard options:

### Option A (most principled): permutations / beta-approx (FastQTL-style)

Use a permutation scheme like those used for cis-eQTL “eGene” mapping: permute the phenotype across samples (or equivalently permute residuals) and recompute the min-p across SNPs to get an empirical null for the *minimum p-value statistic*, then beta-approximate it. tensorQTL implements this style for cis mapping, and provides permutation utilities for trans workflows too; trans mode itself is nominal and sparse by default. ([GitHub][1])

This gives you **one p-value per (A,B)** that already accounts for SNP count and LD in A’s window.

### Option B (faster approximation): eigenMT-style effective tests

Compute an effective number of independent SNP tests in the window (from LD eigenvalues) and do:
[
p^{A\rightarrow B}*{\text{edge}} \approx p*{\min}\times m_{\text{eff}}(A)
]
This is exactly the motivation of eigenMT: correct within a cis window accounting for LD so you don’t just penalize by raw SNP count. ([PMC][2])

**Practical note about gene length:** you generally *don’t* include gene length as a covariate; the bias arises because longer genes (or gene-dense regions) have more SNPs / LD blocks. Handling **within-edge correction LD-aware** largely addresses this. If you skip this step and just use nominal min-p, long genes will spray edges.

---

## 3) Across-edge multiple testing: control FDR on the set of directed edges

If your module has (M) genes, you have up to (M(M-1)) directed hypotheses. After step (2), you have one p-value per directed edge.

Then apply an FDR procedure across all edges, e.g.:

* **Benjamini–Hochberg (BH)** on ({p^{A\rightarrow B}_{\text{edge}}}) to control edge-FDR at (say) 5%.
* If you’re worried about arbitrary dependence among tests, BH is often still used; Benjamini–Yekutieli is more conservative.

A key reason to prefer this **hierarchical** “within-edge then across-edge” approach is that applying FDR directly to all SNP–gene tests can badly mis-control error rates for higher-level calls (like “this gene has an eQTL” or “this edge exists”). This is a known issue; hierarchical procedures are recommended when your scientific claims are at the gene (or edge) level. ([OUP Academic][3])

**Result:** your network edges are the ((A,B)) pairs that pass edge-level q-value < 0.05 (or your chosen threshold).

---

## 4) Things you should include to avoid false edges (trans-eQTL is fragile)

### A) Covariates / confounding control is everything in trans

Include:

* genotype PCs (population structure)
* technical covariates (batch, RIN, etc.)
* expression hidden factors (PEER / expression PCs)
* optionally kinship / LMM if related samples

This is standard practice in large cis/trans eQTL studies; trans is especially sensitive because effects are small and confounding induces massive false positives. ([PMC][4])

### B) Filter “cis leakage” near B

Even if you intend trans, if A’s window overlaps B (or is near B), you’ll rediscover cis and local LD artifacts. tensorQTL’s trans workflow explicitly filters cis by distance to the phenotype TSS in its helper. ([GitHub][1])
For your custom A→B definition, you should similarly **exclude SNPs near B** (e.g., within 1–5 Mb) when calling trans-like edges, or at least label them separately.

### C) Multiple signals / LD clumping

If one LD block in A creates 30 edges, that’s not 30 independent pieces of evidence. Consider reporting:

* the **lead variant per edge**
* and optionally a **clumped lead-variant set per A** (or conditional analysis) so your out-degree isn’t inflated by dense LD.

Large trans resources (e.g., eQTLGen) devote a lot of attention to this “hub SNP” phenomenon. ([PMC][4])

### D) Winner’s curse / reproducibility

If possible: replicate edges in an independent cohort/tissue or via split-sample validation. Trans edges are notorious for poor replication without careful modeling. ([PMC][4])

---

## 5) A conceptual warning: your edges are “locus A → expression B”, not “gene A regulates gene B”

Even if the variants are in/near gene A, the causal regulatory element could be:

* acting on another nearby gene (not A),
* affecting a distal enhancer,
* tagging multiple genes in the locus.

So the directed edge you infer is better interpreted as:

> **“genetic variation near A is associated with expression of B”**

If you truly want **gene A → gene B regulation**, add a mediation step:

* First map **cis-eQTLs for A** (variants near A affecting expression of A).
* For an A-window SNP that hits B, test whether the SNP→B association is **mediated by expression of A** (e.g., SNP→B effect attenuates when conditioning on A expression; or use MR / causal inference tests).
  This is the kind of additional step used in many “variant → gene → downstream gene/pathway” interpretations.

Without mediation, the network is valid as a **directed genetic-association network**, but it’s not guaranteed to be a regulatory gene network.

---

## 6) If you want more power (and fewer tests), consider module-aware trans methods

Your design still has a huge testing burden: (\sum_{A\neq B} |S_A|). Methods that explicitly test **SNP → module / gene-set** effects can be much more powerful than scanning every B separately (especially if the locus affects many module genes). For example, trans-PCO was designed for trans mapping in gene sets / co-expression modules to reduce multiple testing and capture coordinated effects. ([ScienceDirect][5])
You can still convert significant SNP→module signals back into edges afterward, but with better control of burden.

---

# A clean “recommended” pipeline for your network

1. **Preprocess**

* normalize expression; regress covariates + PEER/PCs; keep residuals.

2. **For each directed pair (A,B)**

* test all SNPs in A’s cis window vs expression(B) with the same covariate model.
* compute **edge p-value** using **permutation/beta** (preferred) or **eigenMT**-style effective tests.

3. **Across all (A,B)**

* BH-FDR on edge p-values → q-values.
* call edges at q < 0.05 (or stricter).

4. **Annotate / sanity checks**

* flag edges where SNPs are near B (possible cis).
* LD-clump lead SNPs, report independent signals.
* replicate / validate where possible.
* (optional) mediation to upgrade “locus edge” → “gene-regulatory edge”.

---

If you tell me roughly: **module size (M), sample size (N), tissue, and whether you have genotype PCs/PEER factors already**, I can suggest concrete thresholds (e.g., how many PEER factors, whether to prefer permutations vs eigenMT for your scale) and a way to structure outputs so each edge has: lead SNP, effect size, edge-q, and an LD-clumped signal ID.

[1]: https://github.com/broadinstitute/tensorqtl "GitHub - broadinstitute/tensorqtl: Ultrafast GPU-enabled QTL mapper · GitHub"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4716687/?utm_source=chatgpt.com "An Efficient Multiple-Testing Adjustment for eQTL Studies that ..."
[3]: https://academic.oup.com/nar/article/46/22/e133/5090771?utm_source=chatgpt.com "Power, false discovery rate and Winner's Curse in eQTL studies"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8432599/?utm_source=chatgpt.com "Large-scale cis- and trans-eQTL analyses identify thousands ..."
[5]: https://www.sciencedirect.com/science/article/pii/S2666979X24000946?utm_source=chatgpt.com "Trans-eQTL mapping in gene sets identifies network effects ..."

