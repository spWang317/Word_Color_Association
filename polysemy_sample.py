"""Pre-filter candidate nouns for the polysemy sensitivity analysis.

Draws a deterministic random sample from the Brown corpus restricted to
common English nouns and retains only those that already satisfy the
WordNet conditions required for the downstream analysis:

  (i)  the word has at least two noun synsets (K >= 2), and
  (ii) at least two of those synsets yield a non-empty disambiguator
       under the deterministic selection rule, so that a within-word
       color-vector spread can be computed.

The returned pool is larger than the target sample size so that the
scraping stage can stop once `target_sample_size` words have completed
the pipeline, absorbing any per-query image-retrieval failures without
shrinking the systematic sample.

Output
------
csv/revision_results/polysemy_sample.csv
    columns: word, brown_count, K, valid_senses
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd

OUT_PATH = Path("csv/revision_results/polysemy_sample.csv")


def _clean(lemma_name, word):
    parts = lemma_name.split("_")
    remaining = [p for p in parts if p.lower() != word.lower()]
    if not remaining:
        return None
    return "_".join(remaining)


def pick_disambiguator(lemma_names, query_word):
    cleaned = []
    for ln in lemma_names:
        if ln.lower() == query_word.lower():
            continue
        c = _clean(ln, query_word)
        if c:
            cleaned.append(c)
    if not cleaned:
        return None
    single = [c for c in cleaned if "_" not in c]
    pool = single if single else cleaned
    pool.sort(key=lambda x: (len(x), x.lower()))
    return pool[0]


def passes_filter(word, min_K=2, min_valid_senses=2):
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets(word, pos="n")
    if len(synsets) < min_K:
        return None
    seen = set()
    valid = 0
    for s in synsets:
        disambig = pick_disambiguator(s.lemma_names(), word)
        if disambig is None:
            continue
        query = f"{word} {disambig.replace('_', ' ')}"
        if query in seen:
            continue
        seen.add(query)
        valid += 1
    if valid < min_valid_senses:
        return None
    return len(synsets), valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, default=60,
                        help="Number of candidate words to write to the sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-frequency", type=int, default=20,
                        help="Minimum Brown corpus frequency.")
    parser.add_argument("--min-k", type=int, default=2,
                        help="Minimum WordNet noun sense count.")
    parser.add_argument("--min-valid-senses", type=int, default=2,
                        help="Minimum number of senses with a valid "
                             "disambiguator under the selection rule.")
    args = parser.parse_args()

    nltk.download("brown", quiet=True)
    nltk.download("universal_tagset", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import brown

    counts = Counter(
        word.lower()
        for word, pos in brown.tagged_words(tagset="universal")
        if pos == "NOUN" and word.isalpha() and len(word) >= 3
    )
    base_pool = sorted(w for w, c in counts.items() if c >= args.min_frequency)
    rng = random.Random(args.seed)
    rng.shuffle(base_pool)

    selected = []
    for w in base_pool:
        result = passes_filter(w, args.min_k, args.min_valid_senses)
        if result is None:
            continue
        K, valid = result
        selected.append((w, counts[w], K, valid))
        if len(selected) >= args.pool_size:
            break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected, columns=["word", "brown_count", "K", "valid_senses"]).to_csv(
        OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(selected)} words "
          f"passing K>={args.min_k}, valid_senses>={args.min_valid_senses})")


if __name__ == "__main__":
    main()
