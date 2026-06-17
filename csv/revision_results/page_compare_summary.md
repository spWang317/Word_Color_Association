# R2.6b — Page 1 vs Page 10 (or last reachable) color vector comparison

- Sample: 30 frequency-random common nouns (same as R2.6 v2).
- For each word, page 1 = top viewport at scroll position 0; page N = viewport at scroll position (N-1) × viewport_height after triggering Google's infinite scroll / 'Show more' button as needed.
- N target = 10. If the scroll hits the bottom of results before N=10, the deepest reachable page (max_page) is used instead.

## Category breakdown

| Category | Description | n |
|---|---|---:|
| A | max_page == 10 (target reached) | 28 |
| B | 1 < max_page < 10 (limited — bottom hit early) | 2 |
| C | max_page == 1 (single-page query, cosine ≡ 1 trivially) | 0 |
| FAIL/MISSING | scrape error or missing screenshot | 0 |

## Cosine similarity summary by category

| Category | n | mean cos | SD | min | max |
|---|---:|---:|---:|---:|---:|
| A_full_10 | 28 | 0.9467 | 0.0342 | 0.8707 | 0.9894 |
| B_partial | 2 | 0.8878 | 0.0043 | 0.8847 | 0.8908 |

## Combined A+B (non-trivial: page 1 vs deepest reached, max_page ≥ 2)

- n = 30
- mean cosine(page1, page_N) = 0.9427
- SD = 0.0362
- median = 0.9504
- min = 0.8707  (word: position)

## Per-word table (sorted by cosine ascending)

| Word | max_page | cosine(page1, pageN) | Category |
|---|---:|---:|---|
| position | 10 | 0.8707 | A_full_10 |
| vice | 10 | 0.8712 | A_full_10 |
| situation | 8 | 0.8847 | B_partial |
| nothing | 9 | 0.8908 | B_partial |
| form | 10 | 0.8934 | A_full_10 |
| peace | 10 | 0.9042 | A_full_10 |
| squad | 10 | 0.9083 | A_full_10 |
| customer | 10 | 0.9102 | A_full_10 |
| equivalent | 10 | 0.9245 | A_full_10 |
| activity | 10 | 0.9329 | A_full_10 |
| jail | 10 | 0.9376 | A_full_10 |
| character | 10 | 0.9378 | A_full_10 |
| aside | 10 | 0.9433 | A_full_10 |
| blog | 10 | 0.9461 | A_full_10 |
| business | 10 | 0.9483 | A_full_10 |
| morning | 10 | 0.9525 | A_full_10 |
| surprise | 10 | 0.9538 | A_full_10 |
| week | 10 | 0.9585 | A_full_10 |
| government | 10 | 0.9637 | A_full_10 |
| boyfriend | 10 | 0.9643 | A_full_10 |
| success | 10 | 0.9653 | A_full_10 |
| television | 10 | 0.9687 | A_full_10 |
| door | 10 | 0.9706 | A_full_10 |
| college | 10 | 0.9712 | A_full_10 |
| corporation | 10 | 0.9746 | A_full_10 |
| escape | 10 | 0.9802 | A_full_10 |
| boat | 10 | 0.9877 | A_full_10 |
| study | 10 | 0.9886 | A_full_10 |
| device | 10 | 0.9891 | A_full_10 |
| daughter | 10 | 0.9894 | A_full_10 |