# Third-party data (not included)

The validation scripts use two third-party datasets. They are not redistributed in this
repository; download them from their official sources and place them at the paths below.

## NRC Word–Colour Association Lexicon v0.92 (Mohammad 2011)
- Used by: `nrc_validation.py`
- Official page: http://www.purl.com/net/lexicons (redirects to https://saifmohammad.com/WebPages/lexicons.html).
  The lexicon's terms of use ask that interested parties be directed to this page rather than
  receive redistributed copies.
- Place the sense-level file at:
  `data_external/NRC-Colour-Lexicon-v0.92/NRC-color-lexicon-senselevel-v0.92.txt`

## comp-syn pre-computed vectors (Guilbeault et al. 2020; Srinivasa Desikan et al. 2020)
- Used by: `compsyn_compare.py`
- Source: the comp-syn package, https://github.com/comp-syn/comp-syn — the pre-computed
  `vectors.json` is linked from that repository's README.
- Place the file at: `external_compare/compsyn/compsyn_vectors.json`
