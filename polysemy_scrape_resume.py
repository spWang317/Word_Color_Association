"""Continue scraping until N words have both bare and >=2 sense vectors.

Reads the candidate `polysemy_queries.csv` and the partial
`polysemy_vectors.csv`, then runs the standard `word2image` pipeline on
queries that are still missing. Processing stops once `target_words`
distinct query words have accumulated a bare-token vector and at least
`min_senses` sense-specific vectors -- the minimum needed to compute
the within-word color-vector spread.

Output
------
csv/revision_results/polysemy_vectors.csv         (appended)
csv/revision_results/polysemy_vectors_skipped.csv (appended)
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


def per_word_state(vec_df):
    """Return {word: {'bare': bool, 'senses': int}} from a vectors DataFrame."""
    state = {}
    for w, g in vec_df.groupby("query_word"):
        bare = ((g["synset_id"].fillna("") == "")).sum() > 0
        senses = ((g["synset_id"].fillna("") != "")).sum()
        state[w] = {"bare": bare, "senses": int(senses)}
    return state


def _is_complete(word_state, min_senses):
    return word_state["bare"] and word_state["senses"] >= min_senses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    parser.add_argument("--vectors", default=str(VECTORS_PATH))
    parser.add_argument("--skipped", default=str(SKIPPED_PATH))
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--target-words", type=int, default=30)
    parser.add_argument("--min-senses", type=int, default=2)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    queries = pd.read_csv(args.queries)
    if Path(args.vectors).exists() and Path(args.vectors).stat().st_size > 1:
        existing = pd.read_csv(args.vectors)
    else:
        existing = pd.DataFrame()

    state = per_word_state(existing) if not existing.empty else {}
    done_queries = set(existing["query"]) if not existing.empty else set()

    new_rows = []
    new_skipped = []

    initial_complete = sum(1 for ws in state.values()
                           if _is_complete(ws, args.min_senses))
    print(f"[start] {initial_complete} / {args.target_words} complete words")

    for current_word, group in queries.groupby("query_word", sort=False):
        if sum(1 for ws in state.values() if _is_complete(ws, args.min_senses)) >= args.target_words:
            break
        state.setdefault(current_word, {"bare": False, "senses": 0})
        for _, row in group.iterrows():
            query = row["query"]
            if query in done_queries:
                continue
            try:
                vec = compute_color_vector(query, base_dir)
            except Exception as e:
                print(f"  {query!r} -> ERROR {e!r}")
                new_skipped.append((query, repr(e)))
                continue
            if vec is None:
                print(f"  {query!r} -> SKIPPED (no image)")
                new_skipped.append((query, "no_image"))
                continue
            record = {col: row[col] for col in
                      ("query_word", "synset_id", "sense_index",
                       "disambiguator", "query")}
            for c, v in zip(COLORS, vec):
                record[c] = v
            new_rows.append(record)
            done_queries.add(query)
            is_bare = (pd.isna(row["synset_id"]) or row["synset_id"] == "")
            if is_bare:
                state[current_word]["bare"] = True
            else:
                state[current_word]["senses"] += 1
            print(f"  {query!r} -> done")

        complete = sum(1 for ws in state.values()
                       if _is_complete(ws, args.min_senses))
        print(f"  [progress] {complete} / {args.target_words} complete words")

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_csv(args.vectors, index=False)
        print(f"appended {len(new_rows)} vectors -> {args.vectors}")
    if new_skipped:
        prev = (pd.read_csv(args.skipped)
                if Path(args.skipped).exists() else
                pd.DataFrame(columns=["query", "reason"]))
        sk = pd.concat([prev, pd.DataFrame(new_skipped, columns=["query", "reason"])],
                       ignore_index=True)
        sk.to_csv(args.skipped, index=False)
        print(f"appended {len(new_skipped)} skipped -> {args.skipped}")

    final_complete = [w for w, ws in state.items() if _is_complete(ws, args.min_senses)]
    print(f"Complete words ({len(final_complete)}): {final_complete}")


if __name__ == "__main__":
    main()
