"""Shared loader for the christos-c parallel Bible corpus.

Used by bible_poc.py, bible_romanize.py and the notebook so the language
selection, verse alignment and romanization caching live in one place.
"""
import csv
import os
import re
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "corpus", "bible-corpus")
BIBLES = os.path.join(BASE, "bibles")

# --- Glottolog-consistent corrections to the christos-c metadata.csv ---------
# (per CRITICAL_REVIEW_23jun): the macro-family 'Altaic' is rejected by modern
# consensus, and a literal 'Altaic' vs 'Altaic(?)' typo split Turkish from Korean.
# Fix to current top-level families (Turkic / Koreanic / Japonic as separate
# isol./families — NOT force-merged) and strip '(?)' uncertainty markers so the
# gold tree is reproducible and defensible. These leave 3 singletons in the 57-set,
# so report the resulting purity ceiling (~0.895). Applied in load().
FAMILY_FIXES = {"Altaic": "Turkic", "Altaic(?)": "Koreanic"}
SCRIPT_FIXES = {"Icelandic": "Latin"}   # metadata 'Ethiopic' is a data error (Latin)


def fix_family(fam):
    f = FAMILY_FIXES.get((fam or "").strip(), (fam or "").strip())
    return f.replace("(?)", "").strip() or "NA"

# Diverse multi-family / multi-script subset (excludes PART/partial bibles so the
# common-verse backbone stays large).
SELECTION = [
    "English.xml","German.xml","Dutch.xml","Swedish.xml","Danish.xml","Icelandic.xml",
    "Norwegian.xml","Afrikaans.xml",
    "French.xml","Spanish.xml","Italian.xml","Portuguese.xml","Romanian.xml","Latin.xml",
    "Russian.xml","Polish.xml","Czech.xml","Bulgarian.xml","Serbian.xml","Croatian.xml",
    "Slovak.xml","Slovene.xml","Ukranian-NT.xml",
    "Lithuanian.xml","Latvian-NT.xml",
    "Finnish.xml","Hungarian.xml",
    "Greek.xml","Albanian.xml",
    "Hindi.xml","Farsi.xml","Nepali.xml","Marathi.xml",
    "Kannada.xml","Malayalam.xml","Telugu.xml",
    "Turkish.xml",
    "Hebrew.xml","Arabic.xml","Amharic.xml","Syriac-NT.xml",
    "Chinese.xml","Korean.xml","Japanese.xml","Vietnamese.xml","Thai.xml","Burmese.xml",
    "Indonesian.xml","Tagalog.xml","Cebuano.xml","Malagasy.xml","Maori.xml",
    "Swahili-NT.xml","Xhosa.xml","Shona.xml","Zulu-NT.xml",
    "Basque-NT.xml",
]


def load_meta():
    with open(os.path.join(BASE, "metadata.csv"), newline="", encoding="utf-8") as f:
        return {r["Filename"]: r for r in csv.DictReader(f)}


def parse_verses(path):
    out = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag.split("}")[-1] == "seg" and el.attrib.get("type") == "verse":
            vid = el.attrib.get("id")
            if vid:
                out[vid] = "".join(el.itertext())
            el.clear()
    return out


def safe(label):
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_") or "X"


def load(selection=SELECTION, verse_cap=4000, char_cap=None, align=True,
         return_units=False):
    """Return dict with per-language data.

    align=True (default): strictly PARALLEL — the same `verse_cap` verses common
    to every language (content-controlled; right for a curated set).
    align=False: COMPARABLE — each language uses its own verses (same genre, not
    the identical verses); needed at large breadth where no verse is shared by
    all (e.g. indigenous NT translations with different versification).

    return_units=True (requires align=True): also return units{label->[verse str]},
    the per-language ALIGNED verse list (same order/length across languages), for the
    Felsenstein site/block bootstrap (langtree.branch_support_block / bootstrap_ci_block).

    Family labels are normalised via FAMILY_FIXES / fix_family (Altaic typo + '(?)'
    stripped) and Script via SCRIPT_FIXES, so the gold tree is reproducible/defensible.

    Keys: names, rawtext{label->pre-clean str}, rows[(label,fam,gen,sub)],
    iso{label->ISO639-3}, script{label->script}, common[verse ids], [units]."""
    meta = load_meta()
    present = [fn for fn in selection if os.path.exists(os.path.join(BIBLES, fn))]
    verses = {fn: parse_verses(os.path.join(BIBLES, fn)) for fn in present}

    common = None
    if align:
        for fn in present:
            common = set(verses[fn]) if common is None else (common & set(verses[fn]))
        common = sorted(common)[:verse_cap]

    names, rawtext, rows, iso, script, used = [], {}, [], {}, {}, set()
    units = {}
    for fn in present:
        m = meta[fn]
        lab = safe(m["Language"])
        while lab in used:
            lab += "_"
        used.add(lab)
        if align:
            unit_list = [verses[fn][v] for v in common]
        else:
            vids = sorted(verses[fn])
            if verse_cap:
                vids = vids[:verse_cap]
            unit_list = [verses[fn][v] for v in vids]
        txt = " ".join(unit_list)
        if char_cap:
            txt = txt[:char_cap]
        names.append(lab)
        rawtext[lab] = txt
        if return_units:
            units[lab] = unit_list
        rows.append((lab, fix_family(m["Family"]), m["Genus"], m["Subgenus"]))
        iso[lab] = m["ISO_639-3"]
        script[lab] = SCRIPT_FIXES.get(lab, m["Script"])
    out = dict(names=names, rawtext=rawtext, rows=rows, iso=iso,
               script=script, common=common)
    if return_units:
        out["units"] = units
    return out


def romanize_cached(names, rawtext, iso, tag="default"):
    """Romanize each language's raw text with uroman, caching to disk so re-runs
    are instant. Returns {label -> romanized str}. The cache `tag` should encode
    the input budget so a different char_cap doesn't return stale romanizations."""
    import time
    import uroman
    cache_dir = os.path.join(HERE, "corpus", "romanized", tag)
    os.makedirs(cache_dir, exist_ok=True)
    uro = None
    out = {}
    for k, lab in enumerate(names, 1):
        path = os.path.join(cache_dir, lab + ".txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out[lab] = f.read()
        else:
            if uro is None:
                uro = uroman.Uroman()
            t = time.time()
            r = uro.romanize_string(rawtext[lab], lcode=(iso.get(lab) or None))
            with open(path, "w", encoding="utf-8") as f:
                f.write(r)
            out[lab] = r
            print(f"  romanized {lab} ({k}/{len(names)}) "
                  f"{len(rawtext[lab])}ch in {time.time()-t:.1f}s", flush=True)
    return out
