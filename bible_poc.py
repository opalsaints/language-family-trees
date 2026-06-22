"""
Bible proof-of-concept (Stage 1): does the information theory actually do any
work at Bible scale across many families?

Loads the christos-c parallel Bible corpus (verse-aligned, content-controlled),
selects a diverse multi-family / multi-script subset, and compares four distance
methods by how well their UPGMA tree matches the metadata-derived gold tree:

  1. trigram-JS   (the information-theoretic method)
  2. unigram-JS   (frequencies only, no order)
  3. alphabet-Jaccard  (DUMB BASELINE: which characters are used, no frequencies)
  4. char-shuffle trigram-JS  (NEGATIVE CONTROL: order destroyed)

Headline question: does trigram-JS BEAT the alphabet baseline here?
"""
import csv
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

import langtree as lt

BASE = os.path.join(os.path.dirname(__file__), "corpus", "bible-corpus")
BIBLES = os.path.join(BASE, "bibles")
META = os.path.join(BASE, "metadata.csv")

# Diverse multi-family / multi-script subset (filenames in the corpus). Excludes
# PART (partial) bibles so the common-verse backbone stays large.
SELECTION = [
    # Germanic
    "English.xml", "German.xml", "Dutch.xml", "Swedish.xml", "Danish.xml",
    "Icelandic.xml", "Norwegian.xml", "Afrikaans.xml",
    # Romance
    "French.xml", "Spanish.xml", "Italian.xml", "Portuguese.xml",
    "Romanian.xml", "Latin.xml",
    # Slavic
    "Russian.xml", "Polish.xml", "Czech.xml", "Bulgarian.xml",
    "Serbian.xml", "Croatian.xml", "Slovak.xml", "Slovene.xml", "Ukranian-NT.xml",
    # Baltic
    "Lithuanian.xml", "Latvian-NT.xml",
    # Uralic
    "Finnish.xml", "Hungarian.xml",
    # Hellenic / Albanian
    "Greek.xml", "Albanian.xml",
    # Indo-Iranian
    "Hindi.xml", "Farsi.xml", "Nepali.xml", "Marathi.xml",
    # Dravidian
    "Kannada.xml", "Malayalam.xml", "Telugu.xml",
    # Turkic
    "Turkish.xml",
    # Semitic / Afro-Asiatic
    "Hebrew.xml", "Arabic.xml", "Amharic.xml", "Syriac-NT.xml",
    # East / SE Asian
    "Chinese.xml", "Korean.xml", "Japanese.xml", "Vietnamese.xml", "Thai.xml",
    "Burmese.xml",
    # Austronesian
    "Indonesian.xml", "Tagalog.xml", "Cebuano.xml", "Malagasy.xml", "Maori.xml",
    # Niger-Congo
    "Swahili-NT.xml", "Xhosa.xml", "Shona.xml", "Zulu-NT.xml",
    # isolate
    "Basque-NT.xml",
]

VERSE_CAP = 4000   # take the first N common verses (content-controlled budget)


def load_meta():
    rows = {}
    with open(META, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["Filename"]] = r
    return rows


def parse_verses(path):
    """Return {verse_id: text} for all <seg type='verse'> elements."""
    out = {}
    for _, el in ET.iterparse(path, events=("end",)):
        tag = el.tag.split("}")[-1]
        if tag == "seg" and el.attrib.get("type") == "verse":
            vid = el.attrib.get("id")
            if vid:
                out[vid] = "".join(el.itertext())
            el.clear()
    return out


def main():
    t0 = time.time()
    meta = load_meta()
    present = [fn for fn in SELECTION if os.path.exists(os.path.join(BIBLES, fn))]
    missing = [fn for fn in SELECTION if fn not in present]
    if missing:
        print("WARN missing:", missing)

    print(f"Parsing {len(present)} bibles ...")
    verses = {fn: parse_verses(os.path.join(BIBLES, fn)) for fn in present}

    # common verse backbone across ALL selected languages
    common = None
    for fn in present:
        ids = set(verses[fn])
        common = ids if common is None else (common & ids)
    common = sorted(common)
    print(f"common verses across all = {len(common)}; using first {VERSE_CAP}")
    common = common[:VERSE_CAP]

    # build per-language cleaned text from the SAME verses (content-controlled)
    names, texts, rows = [], [], []
    used = set()
    for fn in present:
        m = meta[fn]
        label = lt._safe(m["Language"])
        while label in used:
            label += "_"
        used.add(label)
        txt = lt.clean(" ".join(verses[fn][v] for v in common))
        names.append(label)
        texts.append(txt)
        rows.append((label, m["Family"], m["Genus"], m["Subgenus"]))

    print("chars/lang: min=%d max=%d" % (min(len(t) for t in texts),
                                         max(len(t) for t in texts)))

    # gold tree from taxonomy
    gold = lt.gold_newick_from_rows(rows)

    # four methods
    methods = {
        "trigram-JS (n=3)": lt.js_matrix(texts, n=3),
        "unigram-JS (n=1)": lt.js_matrix(texts, n=1),
        "alphabet-Jaccard (BASELINE)": lt.alphabet_jaccard_matrix(texts),
        "shuffle trigram-JS (CONTROL)": lt.js_matrix([lt.shuffle_chars(t, 0) for t in texts], n=3),
    }

    print(f"\n{'method':32s}  {'RF/denom':>12s}  {'normRF':>7s}  {'NN-family-purity':>16s}")
    print("-" * 76)
    fam_of = {label: fam for label, fam, _, _ in rows}
    results = {}
    for name, D in methods.items():
        nwk = lt.linkage_to_newick(lt.upgma(D, names), names)
        rf, denom, norm = lt.rf_corrected(nwk, gold)
        # nearest-neighbour family purity
        hits = 0
        for i, lab in enumerate(names):
            order = np.argsort([D[i, j] if j != i else np.inf for j in range(len(names))])
            nn = names[order[0]]
            if fam_of[nn] == fam_of[lab]:
                hits += 1
        purity = hits / len(names)
        results[name] = (rf, denom, norm, purity)
        print(f"{name:32s}  {rf:5d}/{denom:<6d}  {norm:7.3f}  {purity:16.3f}")

    print("-" * 76)
    tri = results["trigram-JS (n=3)"][2]
    alpha = results["alphabet-Jaccard (BASELINE)"][2]
    verdict = ("TRIGRAMS BEAT the alphabet baseline" if tri < alpha - 1e-9
               else "TRIGRAMS TIE/LOSE to the alphabet baseline" if tri <= alpha + 1e-9
               else "alphabet baseline beats trigrams")
    print(f"HEADLINE: trigram-JS normRF={tri:.3f} vs alphabet normRF={alpha:.3f} -> {verdict}")
    print(f"(lower normRF = closer to the true family tree; {len(names)} languages)")

    # ---- dendrogram of the trigram-JS tree, coloured by family ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import dendrogram

    fams = sorted(set(fam_of.values()))
    cmap = plt.get_cmap("tab20")
    fam_color = {f: cmap(i % 20) for i, f in enumerate(fams)}
    D = methods["trigram-JS (n=3)"]
    Z = lt.upgma(D, names)
    fig, ax = plt.subplots(figsize=(11, 0.33 * len(names) + 2))
    dendrogram(Z, labels=names, orientation="right", ax=ax,
               color_threshold=0, above_threshold_color="#999")
    ax.set_title("Parallel Bible corpus — character-trigram Jensen–Shannon tree\n"
                 f"{len(names)} languages, UPGMA, labels coloured by true family")
    ax.set_xlabel("Jensen–Shannon distance (character trigrams)")
    for lbl in ax.get_ymajorticklabels():
        lbl.set_color(fam_color[fam_of[lbl.get_text()]])
        lbl.set_fontsize(8)
    legend = [Patch(facecolor=fam_color[f], label=f) for f in fams]
    ax.legend(handles=legend, loc="lower right", fontsize=6, ncol=2, framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "figures", "bible_poc_trigram_tree.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
