"""
R2.6 v2 — Relaxed-filter resampling for the crop robustness check.

Original filter (v1) restricted to *concrete imageable* WordNet lexnames
(artifact/food/animal/plant/object/substance/body/shape/phenomenon),
which a reviewer could argue selects for "well-behaved" words.

v2 RELAXES this by dropping the lexname restriction. We keep only the
*minimal defensive* filters:
- length >= 4, alphabetic
- not stopwords, color terms, numbers, profanity
- noun-dominant in WordNet (so that "choose" / "fake" don't dominate the sample)
- not a proper noun (handled separately by R2.3 polysemy demo)
- singular base form (drop plurals)

This widens the sampled population to include abstract concepts, social/
relational nouns, and any imageable/non-imageable common noun — testing
robustness across a broader word distribution.

Source: wordfreq (Speer et al.) aggregated English frequency.
"""

from __future__ import annotations

import random

import nltk
import wordfreq
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer

NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "twenty", "thirty", "hundred", "thousand",
    "million", "billion", "dozen", "zero",
}

PROFANITY = {
    "shit", "fuck", "damn", "ass", "crap", "hell", "bitch", "dick",
    "piss", "bastard", "cock", "pussy", "tit", "boob",
}

N_SAMPLE = 30       # v2: 30 fresh, no anchors
SEED = 42
TOP_POOL = 3000     # restrict to the 3000 most frequent words

# v2: NO anchor words. The 3 v1 anchors (flower/love/tomato) are explicitly
# EXCLUDED from the candidate pool so the sample is fully fresh.
EXISTING = ["flower", "love", "tomato"]

# Color terms / synonyms to exclude (Table 1 + base names + brown)
COLOR_WORDS = {
    "red", "carmine", "ruby", "scarlet", "vermilion",
    "orange", "apricot", "marmalade", "orangish", "tangerine",
    "yellow", "gold", "golden", "yellowish", "yellowy",
    "green", "greenery", "greenish", "leafy", "verdant",
    "blue", "azure", "cerulean", "cobalt", "ultramarine",
    "purple", "amethyst", "purplish", "purply", "violet",
    "pink", "blushing", "rose", "rosy",
    "black", "blackish", "gray", "grey", "silver", "silvery",
    "white", "snowy", "milky", "chalk",
    "brown", "tan", "beige", "color", "colour", "colored",
}


def main():
    pool = wordfreq.top_n_list("en", TOP_POOL)
    stop = set(stopwords.words("english"))
    lem = WordNetLemmatizer()

    candidates = []
    for w in pool:
        if not w.isalpha():
            continue
        if len(w) < 4:
            continue
        if w in stop or w in COLOR_WORDS or w in NUMBER_WORDS:
            continue
        if w in EXISTING or w in PROFANITY:
            continue
        # singular base form only
        if lem.lemmatize(w, pos="n") != w:
            continue
        # require the noun reading to be DOMINANT over verb and adjective
        n_noun = len(wn.synsets(w, pos="n"))
        n_verb = len(wn.synsets(w, pos="v"))
        n_adj = len(wn.synsets(w, pos="a")) + len(wn.synsets(w, pos="s"))
        if n_noun == 0 or n_noun < n_verb or n_noun < n_adj:
            continue
        # exclude proper nouns (WordNet instances: Europe, Pacific, ...) —
        # these are handled separately by the R2.3 polysemy demo
        syns = wn.synsets(w, pos="n")
        if syns and syns[0].instance_hypernyms():
            continue
        # *** v2: NO concrete-imageable lexname restriction ***
        candidates.append(w)

    print(f"Candidate common nouns (top {TOP_POOL}, RELAXED filter): {len(candidates)}")

    rng = random.Random(SEED)
    sample = sorted(rng.sample(candidates, N_SAMPLE))
    final = sample  # v2: no anchors

    # Print which lexnames the sample drew
    print(f"\n{N_SAMPLE} frequency-sampled common nouns (seed = {SEED}, v2 RELAXED, fresh):")
    for w in sample:
        syns = wn.synsets(w, pos="n")
        lex = syns[0].lexname() if syns else "?"
        print(f"  {w:15s}  ({lex})")
    print(f"\n=== Final list ({len(final)} words) ===")
    print(", ".join(final))

    with open("csv/revision_results/sampled_words_v2.txt", "w") as f:
        f.write("\n".join(final))
    print("\nSaved: csv/revision_results/sampled_words_v2.txt")
    return final


if __name__ == "__main__":
    main()
