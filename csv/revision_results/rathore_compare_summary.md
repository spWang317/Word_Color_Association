# Rathore et al. (2019) human ratings vs our framework — 18-concept extension

**Two complementary Rathore rating tables combined**:

- *RatingsAllMethods.csv* — 12 fruit concepts × 58 UW-58 colors × 54 participants → mean True Rating.
- *RatingsTestConcepts.csv* — 6 material/recycling concepts (Compost, Glass, Metal, Paper, Plastic, Trash) × 37 BCP-37 colors.

**Color-set → 10-base mapping**: Each row of UW-58 / BCP-37 provides (L, c, h). The same chromatic-threshold + hue/light band rule used by the manuscript's color_vector_from_pixels assigns each color to one of our 10 base colors. UW-58 covers all 7 chromatic categories + 3 achromatic (n_red=3, orange=7, yellow=9, green=9, blue=6, purple=11, pink=8, black=2, grey=2, white=1). BCP-37 covers the same 10 categories with smaller per-color counts.

**Per-concept aggregation**: Human ratings on the native palette are summed into the 10 base-color bins per concept; chromatic block (7) and achromatic block (3) are normalized separately to sum 1 — matching our framework's two-block 10-d output.

**Our framework's vectors**: Standard pipeline on Google Image screenshot of each concept (word2image with disambiguating modifiers where needed: 'orange fruit', 'glass material', 'metal material', 'paper material', 'plastic material').

## Aggregate results

| Metric | All 18 | Fruits 12 | Materials 6 |
|---|---:|---:|---:|
| 10-d cosine | 0.8525 | 0.8515 | 0.8546 |
| Chromatic 7-d cosine | 0.7773 | 0.7679 | 0.7961 |
| Achromatic 3-d cosine | 0.9204 | 0.9309 | 0.8993 |
| Pearson r | 0.6558 | 0.6695 | 0.6286 |
| Spearman ρ | 0.6637 | 0.6602 | 0.6707 |
| Top-1 match rate | 0.3333 | 0.4167 | 0.1667 |

## Per-concept table

| Concept | kind | top (human) | top (ours) | match | cos 10d | cos chrom7 | cos achrom3 | Pearson | Spearman |
|---|---|---|---|:-:|---:|---:|---:|---:|---:|
| Avocado | fruit | black | yellow | ✗ | 0.755 | 0.631 | 0.927 | 0.509 | 0.612 |
| Blueberry | fruit | black | grey | ✗ | 0.849 | 0.716 | 0.948 | 0.557 | 0.479 |
| Cantaloupe | fruit | grey | yellow | ✗ | 0.885 | 0.901 | 0.899 | 0.737 | 0.697 |
| Grapefruit | fruit | grey | white | ✗ | 0.879 | 0.830 | 0.904 | 0.589 | 0.552 |
| Honeydew | fruit | grey | yellow | ✗ | 0.792 | 0.688 | 0.960 | 0.631 | 0.697 |
| Lemon | fruit | grey | yellow | ✗ | 0.861 | 0.873 | 0.899 | 0.823 | 0.891 |
| Lime | fruit | green | green | ✓ | 0.938 | 0.935 | 0.966 | 0.931 | 0.914 |
| Mango | fruit | grey | grey | ✓ | 0.888 | 0.771 | 0.973 | 0.775 | 0.709 |
| Orange | fruit | grey | grey | ✓ | 0.879 | 0.920 | 0.854 | 0.859 | 0.766 |
| Raspberry | fruit | grey | grey | ✓ | 0.791 | 0.530 | 0.949 | 0.475 | 0.394 |
| Strawberry | fruit | black | grey | ✗ | 0.798 | 0.601 | 0.937 | 0.466 | 0.515 |
| Watermelon | fruit | grey | grey | ✓ | 0.901 | 0.819 | 0.955 | 0.681 | 0.697 |
| Compost | material | black | orange | ✗ | 0.830 | 0.784 | 0.879 | 0.585 | 0.697 |
| Glass | material | grey | blue | ✗ | 0.842 | 0.769 | 0.920 | 0.659 | 0.891 |
| Metal | material | black | grey | ✗ | 0.882 | 0.791 | 0.924 | 0.671 | 0.588 |
| Paper | material | grey | white | ✗ | 0.825 | 0.815 | 0.831 | 0.550 | 0.588 |
| Plastic | material | grey | grey | ✓ | 0.965 | 0.907 | 0.999 | 0.881 | 0.794 |
| Trash | material | black | blue | ✗ | 0.783 | 0.711 | 0.843 | 0.425 | 0.467 |