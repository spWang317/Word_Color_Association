"""
Clean the manuscript's plos_bibtex_sample.bib for the PLOS revision.

Operations:
  - Remove unused entries (52 keys never cited).
  - Resolve duplicate keys (6 keys appearing twice — keep the entry with
    more complete metadata: DOI, ISBN, publisher).
  - Remove the broken empty-key entry.
  - Append the 6 new BibTeX entries needed by the revision.
"""

import re
from pathlib import Path

SRC = Path("/Users/qgroup/Desktop/[word-color-association]final_draft/plos_bibtex_sample.bib")
NEW_ENTRIES = Path("/Users/qgroup/Desktop/[word-color-association]final_draft/Revision_Package/new_bibtex_entries.bib")

# Cited keys (from manuscript body + revision insertions). 94 + 7 newly added.
CITED = {
    # original manuscript
    "stokman2000error", "gobe2010emotional", "good2005permutation",
    "ernst2004permutation", "newman1999monte", "lindner2012chocolate",
    "setlur2016linguistic", "guilbeault2020color", "arnold2023distant",
    "lin2013semantically", "mukherjee2024mcolor", "rathore2019estimating",
    "hussein2021pragmatic", "azetsu2021chroma", "gage1999color", "yu2014cross",
    "jit2013exploring", "jin2019influence", "ciotti2018psychology",
    "labrecque2012exciting", "shi2008revision", "Gaurav_Bing_Image_Downloader",
    "mclaren1980cielab", "hamilton2004toward", "fariha2009poetic",
    "hama2020imagism", "singh2006impact", "mohammad2013even", "kim2020lexichrome",
    "mohammad2013colourful", "hutchings2004colour", "harashima2016japanese",
    "lafourcade2014crowdsourcing", "mukhitdinovna2022problem",
    "kartashkova2022colour", "olgacolorUSA", "romanyshyn2022corpus",
    "wadsworth2017evolution", "brenning2023web", "yao2012approach",
    "najork2002high", "kim2012color", "hutchings1997folklore", "hunt2006colour",
    "moretti2013distant", "koriat1993prominence", "han2018amygism",
    "thaggert2010images", "paschos2001perceptually", "bhurchandi2000analytical",
    "ozbal2011comparison", "selenium", "kiros2018illustrative", "yao2017exploiting",
    "zhou2012google", "nieuwenhuysen2018information", "pytesseract", "pillow",
    "duchon1979lanczos", "berlin1991basic", "quinn1988evidence",
    "lindsey2004sunlight", "dictionary1989oxford", "roberts2017oxford", "pandas",
    "bird2009natural", "phillips2014visual", "smith2007overview", "jin2010wisdom",
    "izadinia2014image", "hannak2013measuring", "englehardt2014webprivacy",
    "riegler2014reflects", "o2011logo", "lechner2012color", "bottomley2006interactive",
    "hutchinson1995harlem", "powell1997rhapsodies", "gayle1970harlem",
    "Takahashi2000ComparisonOC",
    # added by revision
    "kayMaffi1999", "hardinMaffi1997", "paoletti2012pink", "pastoureau2001blue",
    "pastoureau2008black", "pastoureau2017red", "stclair2017colors",
    "brinPage1998", "joachims2002", "agichtein2006", "craswellSzummer2007",
    "jingBaluja2008", "garg2018word", "kozlowski2019geometry",
    # truly new (will be added from new_bibtex_entries.bib)
    "bradley1997auc", "handTill2001auc", "srinivasaDesikan2020compsyn",
    "safdar2017jzazbz", "kriegeskorte2008rsa", "google2025imageseo",
}

# Keep for SI Table data sources even though not \cite{}'d in main text.
KEEP_FOR_SI_TABLE = {
    "lowell1921legends", "lowell1918can", "lowell1919pictures",
    "lowell1912dome", "lowell1921sword", "lowell1916men", "gdj1997poet",
    "project-gutenberg",
}

KEEP = CITED | KEEP_FOR_SI_TABLE


def parse_entries(text: str):
    """Parse BibTeX into entry blocks, keyed by their citation key.
    Returns list of (key, block_text) preserving order. Entries with
    empty/missing keys are returned with key = '' so the caller can
    filter them out.
    """
    entries = []
    # Match each @TYPE{KEY,...} block (greedy on body via balanced braces)
    # Simple approach: walk character by character tracking braces.
    i = 0
    n = len(text)
    while i < n:
        # find next @
        at = text.find("@", i)
        if at < 0:
            break
        # find first { after @
        open_brace = text.find("{", at)
        if open_brace < 0:
            break
        # extract key (chars between { and ,)
        comma = text.find(",", open_brace)
        if comma < 0:
            break
        key = text[open_brace + 1:comma].strip()
        # walk braces to find matching close
        depth = 1
        j = open_brace + 1
        while j < n and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth != 0:
            # malformed; bail
            break
        block = text[at:j]
        entries.append((key, block))
        i = j
    return entries


def metadata_score(block: str) -> int:
    """Higher = more complete entry. Used to pick which duplicate to keep."""
    s = 0
    if re.search(r"\bdoi\s*=", block, re.I):
        s += 4
    if re.search(r"\bisbn\s*=", block, re.I):
        s += 3
    if re.search(r"\bpublisher\s*=", block, re.I):
        s += 1
    if re.search(r"\bpages\s*=", block, re.I):
        s += 1
    if re.search(r"\baddress\s*=", block, re.I):
        s += 1
    return s


def main():
    text = SRC.read_text(encoding="utf-8")
    entries = parse_entries(text)
    print(f"Parsed {len(entries)} entries from source .bib.")

    # Drop broken (empty key)
    entries = [(k, b) for (k, b) in entries if k.strip()]

    # Group by key
    by_key: dict[str, list[str]] = {}
    order = []
    for k, b in entries:
        if k not in by_key:
            by_key[k] = []
            order.append(k)
        by_key[k].append(b)

    # Resolve duplicates: keep the one with the highest metadata_score
    deduped: dict[str, str] = {}
    duplicates_resolved = []
    for k in order:
        blocks = by_key[k]
        if len(blocks) == 1:
            deduped[k] = blocks[0]
        else:
            best = max(blocks, key=metadata_score)
            deduped[k] = best
            duplicates_resolved.append((k, len(blocks)))
    print(f"Duplicates resolved: {len(duplicates_resolved)} keys "
          f"({sum(c - 1 for _, c in duplicates_resolved)} extra entries dropped).")
    for k, c in duplicates_resolved:
        print(f"  - {k} ({c}x → 1x)")

    # Filter to KEEP set
    kept = {k: deduped[k] for k in order if k in deduped and k in KEEP}
    dropped_unused = [k for k in order if k in deduped and k not in KEEP]
    print(f"\nKept {len(kept)} entries (cited + KEEP_FOR_SI_TABLE).")
    print(f"Dropped {len(dropped_unused)} unused entries:")
    for k in sorted(dropped_unused):
        print(f"  - {k}")

    # Append new entries (only the 6 truly new ones)
    new_text = NEW_ENTRIES.read_text(encoding="utf-8")
    new_entries = parse_entries(new_text)
    new_keys_added = []
    new_blocks = []
    for k, b in new_entries:
        if not k.strip():
            continue
        if k in kept:
            print(f"  (skip {k}: already in kept)")
            continue
        new_blocks.append(b)
        new_keys_added.append(k)
    print(f"\nAppending {len(new_keys_added)} new entries: {new_keys_added}")

    # Compose final .bib
    out = []
    out.append("% =========================================================================")
    out.append("% plos_bibtex_sample.bib — cleaned for PLOS revision PONE-D-26-04810")
    out.append("% Cleaned 2026-06-06 from prior version.")
    out.append("%")
    out.append("% Operations applied:")
    out.append("%   - Removed 52 unused entries (never \\cite'd in body or revision insertions).")
    out.append("%   - Resolved 6 duplicate keys (kept the entry with more complete metadata).")
    out.append("%   - Dropped 1 broken empty-key entry.")
    out.append("%   - Appended 6 new entries needed for the revision (see end of file).")
    out.append("% =========================================================================")
    out.append("")
    for k in order:
        if k in kept:
            out.append(kept[k])
            out.append("")
    out.append("")
    out.append("% =========================================================================")
    out.append("% NEW ENTRIES added for PONE-D-26-04810 revision (2026-06-06)")
    out.append("% =========================================================================")
    out.append("")
    for b in new_blocks:
        out.append(b)
        out.append("")
    final_text = "\n".join(out).rstrip() + "\n"

    SRC.write_text(final_text, encoding="utf-8")
    print(f"\nSaved cleaned bib: {SRC}")
    print(f"Total entries in final: {len(kept) + len(new_keys_added)}")


if __name__ == "__main__":
    main()
