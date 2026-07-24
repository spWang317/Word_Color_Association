# Setlur full-pipeline reproduction — final results

- Initial random sample: 100 English words (seed=0)
- Stage 1 (lexical, WordNet color-term lookup): 17 words
- Stage 2 (image, Bing top-5 → k-means → W3C nearest): 77 words
- Stage 2 fails (no images returned): 6 words (excluded from analysis)
- Evaluable sample: N = 94 words

- Full-pipeline top-1 agreement with framework: 31/94 (33.0%)
- Per-color one-vs-rest AUC: median 0.632 over 8 defined colors

## Per-color AUC

| color | Setlur positives | AUC |
|---|---|---|
| red | 2 | 0.658 |
| orange | 2 | 0.565 |
| yellow | 7 | 0.635 |
| green | 4 | 0.708 |
| blue | 0 | n/a |
| purple | 3 | 0.725 |
| pink | 0 | n/a |
| black | 19 | 0.571 |
| grey | 50 | 0.448 |
| white | 7 | 0.629 |

## Confusion matrix

| setlur_final_bin   |   red |   orange |   yellow |   green |   blue |   purple |   pink |   black |   grey |   white |
|:-------------------|------:|---------:|---------:|--------:|-------:|---------:|-------:|--------:|-------:|--------:|
| red                |     0 |        0 |        0 |       0 |      0 |        0 |      0 |       0 |      1 |       1 |
| orange             |     0 |        0 |        0 |       0 |      1 |        0 |      0 |       0 |      1 |       0 |
| yellow             |     0 |        0 |        1 |       1 |      0 |        0 |      0 |       1 |      4 |       0 |
| green              |     0 |        0 |        0 |       2 |      1 |        0 |      0 |       0 |      1 |       0 |
| blue               |     0 |        0 |        0 |       0 |      0 |        0 |      0 |       0 |      0 |       0 |
| purple             |     0 |        0 |        0 |       0 |      1 |        0 |      0 |       1 |      1 |       0 |
| pink               |     0 |        0 |        0 |       0 |      0 |        0 |      0 |       0 |      0 |       0 |
| black              |     0 |        0 |        0 |       2 |      5 |        0 |      0 |       3 |      9 |       0 |
| grey               |     0 |        2 |        2 |       3 |     10 |        1 |      0 |       0 |     24 |       8 |
| white              |     0 |        0 |        0 |       0 |      1 |        0 |      0 |       0 |      5 |       1 |