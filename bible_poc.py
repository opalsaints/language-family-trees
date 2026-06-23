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
import biblecorpus as bc

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
        rows.append((label, bc.fix_family(m["Family"]), m["Genus"], m["Subgenus"]))

    print("chars/lang: min=%d max=%d" % (min(len(t) for t in texts),
                                         max(len(t) for t in texts)))

    # gold tree from taxonomy (family labels normalised via bc.fix_family above)
    gold = lt.gold_newick_from_rows(rows)
    fam_of = {label: fam for label, fam, _, _ in rows}
    gen_of = {label: (gen or fam) for label, fam, gen, _ in rows}

    # four methods
    methods = {
        "trigram-JS (n=3)": lt.js_matrix(texts, n=3),
        "unigram-JS (n=1)": lt.js_matrix(texts, n=1),
        "alphabet-Jaccard (BASELINE)": lt.alphabet_jaccard_matrix(texts),
        "shuffle trigram-JS (CONTROL)": lt.js_matrix([lt.shuffle_chars(t, 0) for t in texts], n=3),
    }

    # honest baselines for the scores
    fam_chance = lt.purity_chance_floor(fam_of, names)
    gen_chance = lt.purity_chance_floor(gen_of, names)
    print(f"\nHONEST BASELINES: family-NN chance floor = {fam_chance:.3f} (NOT 0; "
          f"largest family = {max(np.unique([f for f in fam_of.values()], return_counts=True)[1])}/{len(names)}); "
          f"genus-NN chance floor = {gen_chance:.3f}")
    print(f"{'method':30s}{'normRF':>8s}{'(floor':>7s}{'null)':>6s}{'rescaled':>9s}{'GQD':>7s}{'fam-NN':>8s}{'ties':>6s}")
    print("-" * 82)
    results = {}
    for name, D in methods.items():
        nwk = lt.linkage_to_newick(lt.upgma(D, names), names)
        tri3 = lt.rf_triple(nwk, gold, names, n_null=200)
        g = lt.gqd(nwk, gold, names)["gqd"]
        diag = lt.nn_diagnostics(D, names, fam_of)
        results[name] = dict(rf=tri3, gqd=g, diag=diag)
        print(f"{name:30s}{tri3['observed']:8.3f}{tri3['floor']:7.3f}{tri3['null_p50']:6.3f}"
              f"{tri3['rescaled']:9.3f}{g:7.3f}{diag['purity']:8.3f}{diag['n_ties']:6d}")
    print("-" * 82)

    tri = results["trigram-JS (n=3)"]
    alpha = results["alphabet-Jaccard (BASELINE)"]
    uni = results["unigram-JS (n=1)"]
    better = tri["rf"]["observed"] < alpha["rf"]["observed"] - 1e-9
    print(f"HEADLINE (vs the DUMB baseline, on the true scale): trigram-JS rescaledRF "
          f"{tri['rf']['rescaled']:.3f} / GQD {tri['gqd']:.3f} / fam-NN {tri['diag']['purity']:.3f}  vs  "
          f"alphabet rescaledRF {alpha['rf']['rescaled']:.3f} / GQD {alpha['gqd']:.3f} / fam-NN {alpha['diag']['purity']:.3f}"
          f"  -> {'trigram-JS WINS' if better else 'NOT better'}")
    print(f"  honest margins: fam-NN +{tri['diag']['purity']-alpha['diag']['purity']:.3f} over baseline, "
          f"+{tri['diag']['purity']-fam_chance:.3f} over CHANCE ({fam_chance:.3f}).")
    print(f"  unigram-JS fam-NN {uni['diag']['purity']:.3f} vs trigram {tri['diag']['purity']:.3f} "
          f"-> order adds {tri['diag']['purity']-uni['diag']['purity']:+.3f} (most signal is in the frequencies).")
    print(f"  tie-robust fam-NN (JS=1.0 ties counted as misses): trigram {tri['diag']['purity_tiemiss']:.3f} "
          f"({tri['diag']['n_ties']} undefined NN on raw cross-script).")

    # genus-level + IE-split purity (the honest 'does it recover families' test)
    Dtri = methods["trigram-JS (n=3)"]
    gen_pur = lt.nn_purity(Dtri, names, gen_of)
    ie = [n for n in names if fam_of[n] == "Indo-European"]
    nonie = [n for n in names if fam_of[n] != "Indo-European"]
    def subset_purity(sub):
        if len(sub) < 2:
            return float("nan")
        idx = [names.index(s) for s in sub]
        Dsub = Dtri[np.ix_(idx, idx)]
        return lt.nn_purity(Dsub, sub, fam_of)
    print(f"  genus-level fam... genus-NN purity {gen_pur:.3f} (genus chance {gen_chance:.3f}); "
          f"IE-only fam-NN {subset_purity(ie):.3f} (n={len(ie)}) vs non-IE-only {subset_purity(nonie):.3f} "
          f"(n={len(nonie)}) -> the panel is IE-dominated; non-IE recovery is the harder, honest test.")

    # vocabulary-size bias diagnostic
    Ks = np.array([len(lt.ngram_counter(t, 3)) for t in texts])
    iu = np.triu_indices(len(names), 1)
    maxK = np.array([[max(Ks[i], Ks[j]) for j in range(len(names))] for i in range(len(names))])[iu]
    print(f"  bias check: corr(trigram-JS distance, max #trigrams K) = "
          f"{np.corrcoef(Dtri[iu], maxK)[0,1]:.3f} (plug-in JS partly tracks vocabulary size).")

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
