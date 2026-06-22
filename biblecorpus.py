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


def load(selection=SELECTION, verse_cap=4000, char_cap=None):
    """Return dict with parallel content-controlled per-language data.

    Keys: names, rawtext{label->pre-clean str}, rows[(label,fam,gen,sub)],
    iso{label->ISO639-3}, script{label->script}, common[verse ids].
    `char_cap` (optional) truncates each language's raw text (bounds romanization
    time while keeping the same verses across languages)."""
    meta = load_meta()
    present = [fn for fn in selection if os.path.exists(os.path.join(BIBLES, fn))]
    verses = {fn: parse_verses(os.path.join(BIBLES, fn)) for fn in present}
    common = None
    for fn in present:
        common = set(verses[fn]) if common is None else (common & set(verses[fn]))
    common = sorted(common)[:verse_cap]

    names, rawtext, rows, iso, script, used = [], {}, [], {}, {}, set()
    for fn in present:
        m = meta[fn]
        lab = safe(m["Language"])
        while lab in used:
            lab += "_"
        used.add(lab)
        txt = " ".join(verses[fn][v] for v in common)
        if char_cap:
            txt = txt[:char_cap]
        names.append(lab)
        rawtext[lab] = txt
        rows.append((lab, m["Family"], m["Genus"], m["Subgenus"]))
        iso[lab] = m["ISO_639-3"]
        script[lab] = m["Script"]
    return dict(names=names, rawtext=rawtext, rows=rows, iso=iso,
                script=script, common=common)


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
