"""Generate per-sense disambiguated queries from WordNet noun synsets.

For each input word `w`, retrieves the noun synsets via
`wordnet.synsets(w, pos='n')`. Each synset's `lemma_names()` provides the
synonyms within that sense; one of these is selected as a disambiguator
and concatenated with `w` to form a sense-specific query.

Disambiguator selection
-----------------------
1. Discard any lemma name equal to the query word (case-insensitive).
2. Strip the query word from compound lemma names (underscore-joined
   tokens); e.g., for query word `chip`, the lemma `poker_chip` becomes
   `poker`. A lemma that consists solely of the query word component is
   removed.
3. Among the cleaned candidates, prefer single-word lemmas (no
   underscore). If none remain, retain compound lemmas.
4. Among the surviving pool, choose the shortest lemma; break ties
   alphabetically.

A synset whose `lemma_names()` yields no usable disambiguator is skipped.

Output
------
csv/revision_results/polysemy_queries.csv
    columns: query_word, synset_id, sense_index, disambiguator, query
    The first row per word has `synset_id` empty and represents the
    bare-token query.
"""

import argparse
from pathlib import Path

import nltk
import pandas as pd

SAMPLE_PATH = Path("csv/revision_results/polysemy_sample.csv")
OUT_PATH = Path("csv/revision_results/polysemy_queries.csv")


def clean_disambiguator(lemma_name, query_word):
    """Remove query-word components from a compound lemma name.

    `lemma_name` is the underscore-joined lemma string as returned by
    `synset.lemma_names()`. Returns the cleaned string with the query
    word stripped, or `None` if nothing remains.
    """
    parts = lemma_name.split("_")
    remaining = [p for p in parts if p.lower() != query_word.lower()]
    if not remaining:
        return None
    return "_".join(remaining)


def pick_disambiguator(lemma_names, query_word):
    """Select one disambiguator string for a synset."""
    cleaned = []
    for ln in lemma_names:
        if ln.lower() == query_word.lower():
            continue
        c = clean_disambiguator(ln, query_word)
        if c:
            cleaned.append(c)
    if not cleaned:
        return None
    single = [c for c in cleaned if "_" not in c]
    pool = single if single else cleaned
    pool.sort(key=lambda x: (len(x), x.lower()))
    return pool[0]


def queries_for_word(query_word):
    """List of (synset_id, sense_index, disambiguator, query) tuples.

    The first tuple is the bare-token query with `synset_id` and
    `disambiguator` set to `None`. Senses whose disambiguator yields a
    query string already emitted for an earlier sense are skipped, so
    the returned list has unique query strings.
    """
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    out = [(None, 0, None, query_word)]
    seen_queries = {query_word}
    for idx, synset in enumerate(wn.synsets(query_word, pos="n"), start=1):
        disambig = pick_disambiguator(synset.lemma_names(), query_word)
        if disambig is None:
            continue
        query = f"{query_word} {disambig.replace('_', ' ')}"
        if query in seen_queries:
            continue
        seen_queries.add(query)
        out.append((synset.name(), idx, disambig, query))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=str(SAMPLE_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    sample_df = pd.read_csv(args.sample)
    rows = []
    for word in sample_df["word"]:
        for synset_id, sense_idx, disambig, query in queries_for_word(word):
            rows.append({
                "query_word": word,
                "synset_id": synset_id or "",
                "sense_index": sense_idx,
                "disambiguator": disambig or "",
                "query": query,
            })
    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    n_words = out_df["query_word"].nunique()
    n_queries = len(out_df)
    print(f"Wrote {args.out} ({n_words} words, {n_queries} queries)")


if __name__ == "__main__":
    main()
