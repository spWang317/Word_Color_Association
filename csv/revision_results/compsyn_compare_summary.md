# comp-syn (Guilbeault et al. 2020; Srinivasa Desikan et al. COLING 2020) vs our framework — representational comparison

**comp-syn** is an independent project that derives per-word color embeddings from Google Image search results via a 3D color-space histogram (2×2×2 = 8 bins) computed in the perceptually uniform JzAzBz space (and analogously in RGB). It releases pre-computed vectors for 39,949 English words from github.com/comp-syn/comp-syn.

**Our framework** outputs a 10-d color ratio vector per word (7 chromatic + 3 achromatic categories in CIELch). We build a per-word lookup over 9,244 English content words by averaging the per-occurrence ratios from the processed corpus.

**Overlap**: 5,449 words appear in both vocabularies.

## Method: Representational Similarity Analysis (RSA)

Because the two embeddings live in different geometries (10-d categorical bins vs 8-d cube histograms), individual dimensions are not directly alignable. RSA compares the *global similarity structure* of the two spaces: for a sample of N words, build the N×N pairwise-cosine matrix in each space, take the upper triangle, and correlate the two long vectors of word-pair similarities. The result is scale-invariant and dimensionality-independent.

- N = 100 words per draw (uniformly random subsample of the 5,449-word overlap).
- K = 100 independent random draws (seeds 0..99).
- Pair count per draw = 4,950.

## Results (distribution across K random draws)

| Comparison | Pearson r (median) | Pearson r (mean ± SD) | Spearman ρ (median) | Spearman ρ (mean ± SD) |
|---|---:|---:|---:|---:|
| Ours (10-d) ↔ comp-syn JzAzBz (8-d) | 0.4581 | 0.4596 ± 0.0575 | 0.4211 | 0.4220 ± 0.0531 |
| Ours (10-d) ↔ comp-syn RGB (8-d) | 0.4594 | 0.4494 ± 0.0578 | 0.4239 | 0.4201 ± 0.0507 |

## Permutation null (100 row-shuffles on a representative draw)

- null Pearson r: mean = -0.0048, SD = 0.0424, max = 0.1186.
- Observed Pearson r exceeds the null max in **100/100** draws.

## Sanity check — top-8 nearest neighbors of seed words

Because the two embeddings differ in dimensionality and binning, fine-grained ranked neighbors are not expected to coincide. The Jaccard overlap between the two top-8 lists is therefore small even though the global structure correlates.

| Seed | Jaccard overlap (top-8) |
|---|---:|
| blue | 0/8 |
| rose | 0/8 |
| gold | 0/8 |
| sea | 1/8 |
| stone | 0/8 |