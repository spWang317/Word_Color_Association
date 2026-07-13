"""Run the framework pipeline on each polysemy query and store color vectors.

Reads `csv/revision_results/polysemy_queries.csv` and invokes the standard
`word2image` pipeline for every query (Google Image Search screenshot
gated by the Bing image-count check used throughout the manuscript). The
resulting 10-dimensional CIELch base-color ratio vector is appended to
`csv/revision_results/polysemy_vectors.csv` and per-query failures are
recorded in `csv/revision_results/polysemy_vectors_skipped.csv`.

Processing stops once `target_words` distinct query words have
accumulated both a bare-token vector and at least two sense-specific
vectors, which is the minimum needed for the within-word spread metric.

Output
------
csv/revision_results/polysemy_vectors.csv
    columns: query_word, synset_id, sense_index, disambiguator, query,
             red, orange, yellow, green, blue, purple, pink,
             black, grey, white
csv/revision_results/polysemy_vectors_skipped.csv
    columns: query, reason
"""

import argparse
from pathlib import Path

import pandas as pd

import calculation as cal
import word2image as w2i

QUERIES_PATH = Path("csv/revision_results/polysemy_queries.csv")
VECTORS_PATH = Path("csv/revision_results/polysemy_vectors.csv")
SKIPPED_PATH = Path("csv/revision_results/polysemy_vectors_skipped.csv")
PIXEL_COUNT = 10000
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink",
          "black", "grey", "white"]


def compute_color_vector(query, base_dir):
    """Return the 10-d named base-color ratio vector for `query`, or None."""
    original_dir = base_dir / "Images" / "original_images"
    temp_dir = base_dir / "Images" / "temp"
    resize_dir = base_dir / "Images" / "resize_images"
    w2i.word2image(query, PIXEL_COUNT,
                   str(original_dir), str(temp_dir), str(resize_dir),
                   check_text=True)
    if not (resize_dir / f"{query}.png").exists():
        return None
    h = cal.extracthue(str(resize_dir), query)
    l = cal.extractlightness(str(resize_dir), query)
    if h is None or l is None or len(h) == 0 or len(l) == 0:
        return None
    ratio, _, _ = cal.color_ratio_for_each_term(h, l)
    return [ratio.get(c, 0.0) for c in COLORS]


def _word_is_complete(word_state, min_senses):
    """Return True when a word has both a bare vector and >= min_senses senses."""
    return word_state["bare"] and word_state["senses"] >= min_senses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    parser.add_argument("--out", default=str(VECTORS_PATH))
    parser.add_argument("--skipped-out", default=str(SKIPPED_PATH))
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--target-words", type=int, default=30,
                        help="Stop processing once this many words have both "
                             "a bare vector and at least --min-senses sense "
                             "vectors.")
    parser.add_argument("--min-senses", type=int, default=1,
                        help="Sense-vector count required for a word to be "
                             "considered complete.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    queries_df = pd.read_csv(args.queries)

    state = {}
    rows = []
    skipped = []
    stop = False

    for current_word, group in queries_df.groupby("query_word", sort=False):
        if stop:
            break
        state.setdefault(current_word, {"bare": False, "senses": 0})
        for _, row in group.iterrows():
            try:
                vec = compute_color_vector(row["query"], base_dir)
            except Exception as e:
                print(f"  {row['query']!r} -> ERROR {e!r}")
                skipped.append((row["query"], repr(e)))
                continue
            if vec is None:
                print(f"  {row['query']!r} -> SKIPPED (no image)")
                skipped.append((row["query"], "no_image"))
                continue
            record = {col: row[col] for col in
                      ("query_word", "synset_id", "sense_index",
                       "disambiguator", "query")}
            for c, v in zip(COLORS, vec):
                record[c] = v
            rows.append(record)
            is_bare = (pd.isna(row["synset_id"]) or row["synset_id"] == "")
            if is_bare:
                state[current_word]["bare"] = True
            else:
                state[current_word]["senses"] += 1
            print(f"  {row['query']!r} -> done")

        complete = sum(1 for ws in state.values()
                       if _word_is_complete(ws, args.min_senses))
        print(f"  [progress] {complete} / {args.target_words} complete words")
        if complete >= args.target_words:
            stop = True

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(rows)} vectors)")
    if skipped:
        pd.DataFrame(skipped, columns=["query", "reason"]).to_csv(
            args.skipped_out, index=False)
        print(f"Wrote {args.skipped_out} ({len(skipped)} skipped)")

    complete_words = [w for w, ws in state.items()
                      if _word_is_complete(ws, args.min_senses)]
    print(f"Complete words ({len(complete_words)}): {complete_words}")


if __name__ == "__main__":
    main()
