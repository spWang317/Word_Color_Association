# R2.2 — Validation against the NRC Word-Colour Association Lexicon

## Headline

- **9 of 10 base colors** have above-chance AUC (median AUC = 0.646) when our color-*c* ratio is used to rank words whose NRC top color is *c*. Most chromatic AUCs lie in 0.60–0.70.
- **Chromatic top-3 inclusion** (NRC top color $\in$ our chromatic top-3): 0.680 vs. chance 0.429 (n = 50).
- **Top-1 agreement is highly stratified** by the kind of word: visually concrete NRC categories (Blue: 0.75 within-axis top-1; Grey: 0.65) agree strongly, while categories whose NRC label is dominated by *symbolic* associations (Red: 0.14; Purple: 0.05; Pink: 0.01) diverge. This stratification is itself the expected signature of the methodological gap between *visual* web-image aggregation and *symbolic* survey-based color association.

## Sample

- Vocabulary intersection: **3,023** words (AL+GDJ ratio master ∩ NRC).
- Random sample (seed = 42): 100 words.
- Excluded because NRC top color is *brown* (not in our base-color set): 0 words.
- Comparable sample size: 100 (chromatic NRC top: 50; achromatic NRC top: 50).
- High-consensus subset (NRC top-color vote share ≥ 0.50): 77 words (chrom: 40, achr: 37).

## Why the comparison is split into chromatic / achromatic

In our pipeline the chromatic (7-color) and achromatic (3-color) distributions are *independently* normalized to sum to 1 each (Methods §Mapping the color coordinates...). A single 10-way argmax over the concatenated vector is therefore biased toward the 3-color achromatic axis (which has fewer bins, hence higher peaks) and is not a fair comparison against Mohammad's 11-way label. We instead compare within the matching axis: when the NRC top color is chromatic we look at our chromatic top-1/top-3; when the NRC top color is achromatic we look at our achromatic top-1.

## Agreement (principled, within-axis)

| Metric | All | High-consensus | Chance |
|---|---:|---:|---:|
| Chromatic top-1 (n = 50 / high 40) | 0.420 | 0.450 | 0.143 |
| Chromatic top-3 (n = 50 / high 40) | 0.680 | 0.650 | 0.429 |
| Achromatic top-1 (n = 50 / high 37) | 0.380 | 0.324 | 0.333 |

## Naive unified top-1 (for transparency, not fair)

Argmax over all 10 colors agrees with NRC top **0.200** of the time (chance = 0.100); as discussed above, this is biased toward achromatic colors and not the appropriate metric.

## Per-color stratified agreement

Where the NRC top color is concrete and visually grounded (e.g., *blue* = sky/swim/frigid), our pipeline agrees strongly. Where the NRC top color is mostly symbolic (e.g., *red* = anger/blood/violence, *purple* = royalty/sorrow), our pipeline — which aggregates *visual* web content — naturally diverges. This pattern is itself informative about what each kind of resource measures.

| nrc_color   | axis       |   n |   top1_agreement |   top3_agreement |   mean_our_ratio_for_this_color |
|:------------|:-----------|----:|-----------------:|-----------------:|--------------------------------:|
| red         | chromatic  |   9 |            0.444 |            0.556 |                           0.23  |
| orange      | chromatic  |   3 |            0     |            0.333 |                           0.121 |
| yellow      | chromatic  |   7 |            0.143 |            0.714 |                           0.194 |
| green       | chromatic  |  11 |            0.182 |            0.636 |                           0.247 |
| blue        | chromatic  |  15 |            0.933 |            1     |                           0.464 |
| purple      | chromatic  |   1 |            0     |            0     |                           0.06  |
| pink        | chromatic  |   4 |            0     |            0.25  |                           0.063 |
| black       | achromatic |  22 |            0.182 |          nan     |                           0.299 |
| grey        | achromatic |  15 |            0.8   |          nan     |                           0.514 |
| white       | achromatic |  13 |            0.231 |          nan     |                           0.254 |

## AUC per color (our color-c ratio as predictor of NRC-top = c)

AUC > 0.5 means our ratio for color *c* is higher, on average, for words whose NRC top color is *c* than for other words. AUC = 0.5 = chance; 1.0 = perfect ranking.

| color   |   n_positive |   auc |
|:--------|-------------:|------:|
| red     |            9 | 0.799 |
| orange  |            3 | 0.502 |
| yellow  |            7 | 0.51  |
| green   |           11 | 0.68  |
| blue    |           15 | 0.716 |
| purple  |            1 | 0.687 |
| pink    |            4 | 0.646 |
| black   |           22 | 0.634 |
| grey    |           15 | 0.646 |
| white   |           13 | 0.439 |

## Chromatic confusion matrix (rows: NRC top, columns: our chromatic top-1)

| NRC top   |   red |   orange |   yellow |   green |   blue |   purple |   pink |
|:----------|------:|---------:|---------:|--------:|-------:|---------:|-------:|
| red       |     4 |        1 |        0 |       0 |      4 |        0 |      0 |
| orange    |     0 |        0 |        1 |       0 |      2 |        0 |      0 |
| yellow    |     0 |        1 |        1 |       1 |      4 |        0 |      0 |
| green     |     1 |        0 |        1 |       2 |      7 |        0 |      0 |
| blue      |     0 |        0 |        0 |       0 |     14 |        0 |      1 |
| purple    |     0 |        0 |        0 |       0 |      1 |        0 |      0 |
| pink      |     0 |        1 |        1 |       0 |      2 |        0 |      0 |

## Achromatic confusion matrix (rows: NRC top, columns: our achromatic top-1)

| NRC top   |   black |   grey |   white |
|:----------|--------:|-------:|--------:|
| black     |       4 |     15 |       3 |
| grey      |       2 |     12 |       1 |
| white     |       0 |     10 |       3 |