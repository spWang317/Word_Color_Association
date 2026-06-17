# R2.4 — Permutation Null Analysis

**Procedure.** Pool all content tokens from Lowell + Johnson (combined N = 50,809 tokens, 9,244 unique words). Randomly partition the combined token list into two corpora matched to the original sizes (N_AL = 47,396, N_GJ = 3,413). Compute Δ_color for each partition. Repeat B = 10,000 times.

**Interpretation.** If the observed Δ_color lies far in the tail of the null distribution (low *p_two_sided*, large |*z*|), the Lowell–Johnson contrast cannot be explained by random partition of the combined lexicon and is therefore systematic.

## Results

| color   |   observed_delta |   null_mean |   null_std |   z_score (Δ / null_SD) |   percentile_one_sided |   p_two_sided |
|:--------|-----------------:|------------:|-----------:|------------------------:|-----------------------:|--------------:|
| red     |         -0.00382 |      -1e-05 |    0.00194 |                -1.96886 |                   2.62 |        0.0492 |
| orange  |          0.00017 |       1e-05 |    0.0023  |                 0.07474 |                  52.63 |        0.9416 |
| yellow  |          0.01151 |       2e-05 |    0.00235 |                 4.90834 |                 100    |        0      |
| green   |          0.00948 |       0     |    0.00264 |                 3.5955  |                  99.97 |        0.0008 |
| blue    |         -0.01306 |      -2e-05 |    0.00376 |                -3.47601 |                   0.03 |        0.0005 |
| purple  |         -0.00273 |      -0     |    0.00118 |                -2.31391 |                   0.96 |        0.0194 |
| pink    |         -0.00155 |      -1e-05 |    0.00081 |                -1.90401 |                   3.09 |        0.0562 |
| black   |         -0.02482 |      -1e-05 |    0.00247 |               -10.0456  |                   0    |        0      |
| grey    |          0.00563 |      -2e-05 |    0.00215 |                 2.61597 |                  99.55 |        0.0096 |
| white   |          0.01919 |       3e-05 |    0.0031  |                 6.19609 |                 100    |        0      |

**Headline.** All high-confidence colors are extreme tails of the null distribution (|z| > 2.5, p_two_sided < 0.05). Black and White are the most extreme (|z| > 6).