"""IPA arm (epitran) — the principled cross-script representation.

uroman gives a *Latin* transliteration but THROWS AWAY phonological vowels for
abjad/abugida scripts (Arabic/Hebrew are written consonant-skeletons; Brahmic
scripts hang vowels off consonants). epitran is a real grapheme->IPA transducer:
it RESTORES the vowels and puts every language on ONE phonetic alphabet (IPA),
so the cross-script n-gram comparison is over actual sounds, not orthography.

Pipeline:
  1. map each bc.SELECTION language (ISO 639-3 + script) -> an epitran code
     (eng-Latn, deu-Latn, rus-Cyrl, ara-Arab, hin-Deva, tel-Telu, ...).
     epitran has NO map for many of our languages -> graceful-skip + LOG.
  2. transcribe each Bible text (verse_cap=2000, char_cap=30000 to match the
     uroman budget) to IPA, CACHE to corpus/ipa/<tag>/<lab>.txt.
  3. trigram Jensen-Shannon on the IPA strings:
        - nn_diagnostics (family-NN purity + chance floor + tie count),
        - rf_triple AND gqd vs the Glottolog/metadata gold tree,
        - IPA-vs-uroman head-to-head on the cross-script test pairs.
  4. figures/ipa_tree.png coloured by family.

We report on the SAME language set for IPA and uroman (the IPA-covered subset),
so the comparison is apples-to-apples.
"""
import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import langtree as lt
import biblecorpus as bc

warnings.filterwarnings("ignore")
try:
    import epitran.logger
    epitran.logger.logger.setLevel("ERROR")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "v2000_c30000"               # same budget tag as the uroman cache
IPA_DIR = os.path.join(HERE, "corpus", "ipa", TAG)
os.makedirs(IPA_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)

# --- bc.SELECTION label -> epitran code (ISO 639-3 + script aware) ------------
# Codes verified to exist in epitran/data/map. Languages with NO epitran map
# (and no defensible same-language substitute) are left out -> graceful-skip.
EPITRAN_CODE = {
    # Germanic (Latin)
    "English": "eng-Latn",          # needs flite lex_lookup; skipped if absent
    "German": "deu-Latn", "Dutch": "nld-Latn", "Swedish": "swe-Latn",
    "Norwegian": "nno-Latn", "Afrikaans": "afr-Latn",
    # Romance (Latin)
    "French": "fra-Latn", "Spanish": "spa-Latn", "Italian": "ita-Latn",
    "Portuguese": "por-Latn", "Romanian": "ron-Latn",
    # Slavic
    "Russian": "rus-Cyrl", "Polish": "pol-Latn", "Czech": "ces-Latn",
    "Serbian": "srp-Latn", "Croatian": "hrv-Latn", "Slovene": "slv-Latn",
    "Ukranian": "ukr-Cyrl",
    # Baltic
    "Lithuanian": "lit-Latn", "Latvian": "lav-Latn",
    # Uralic
    "Finnish": "fin-Latn", "Hungarian": "hun-Latn",
    # Albanian
    "Albanian": "sqi-Latn",
    # Indo-Iranian
    "Hindi": "hin-Deva", "Farsi_Persian": "fas-Arab", "Marathi": "mar-Deva",
    # Dravidian
    "Kannada": "kan-Knda", "Malayalam": "mal-Mlym", "Telugu": "tel-Telu",
    # Turkic
    "Turkish": "tur-Latn",
    # Semitic
    "Arabic": "ara-Arab", "Amharic": "amh-Ethi",
    # East/SE Asian
    "Korean": "kor-Hang", "Japanese": "jpn-Hira", "Vietnamese": "vie-Latn",
    "Thai": "tha-Thai", "Burmese": "mya-Mymr",
    # Austronesian
    "Indonesian": "ind-Latn", "Tagalog": "tgl-Latn", "Cebuano": "ceb-Latn",
    "Maori": "mri-Latn",
    # Niger-Congo (Bantu)
    "Swahili": "swa-Latn", "Xhosa": "xho-Latn", "Shona": "sna-Latn",
    "Zulu": "zul-Latn",
}
# Deliberately ABSENT (logged below): Danish, Icelandic (no Germanic map);
# Latin (no lat map); Bulgarian (no bul map), Slovak (no slk map);
# Nepali (no nep map); Greek (no ell/grc map); Hebrew (no heb map);
# Syriac (no syr map); Mandarin_Chinese (no Hanzi->IPA map); Malagasy (no plt
# map); Basque (no eus map). English skips at runtime if flite is missing.


def epitran_map_present(code):
    import epitran
    mapdir = os.path.join(os.path.dirname(epitran.__file__), "data", "map")
    return os.path.exists(os.path.join(mapdir, code + ".csv"))


def transcribe_cached(names, rawtext, iso, script):
    """IPA-transcribe each language with epitran, caching to corpus/ipa/<tag>/.
    Returns (ipa{lab->str}, kept[labels], skipped[(lab, reason)])."""
    import epitran
    ipa, kept, skipped = {}, [], []
    eng = {}  # cache Epitran objects per code
    for k, lab in enumerate(names, 1):
        path = os.path.join(IPA_DIR, lab + ".txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                ipa[lab] = f.read()
            kept.append(lab)
            continue
        code = EPITRAN_CODE.get(lab)
        if code is None:
            skipped.append((lab, f"no epitran map (iso={iso.get(lab)}, "
                                 f"script={script.get(lab)})"))
            continue
        if not epitran_map_present(code):
            skipped.append((lab, f"epitran code {code} not installed"))
            continue
        try:
            if code not in eng:
                eng[code] = epitran.Epitran(code)
            t = time.time()
            out = eng[code].transliterate(rawtext[lab])
            # detect a no-op (epitran returned the input unchanged -> failed)
            if out.strip() == rawtext[lab].strip() or not out.strip():
                skipped.append((lab, f"epitran {code} produced no transcription "
                                     f"(missing backend, e.g. flite for eng)"))
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(out)
            ipa[lab] = out
            kept.append(lab)
            print(f"  IPA {lab:16s} ({k}/{len(names)}) {code:9s} "
                  f"{len(rawtext[lab])}ch -> {len(out)}ch in {time.time()-t:.1f}s",
                  flush=True)
        except Exception as ex:
            skipped.append((lab, f"epitran {code} error: {str(ex)[:60]}"))
    return ipa, kept, skipped


def main():
    t0 = time.time()
    d = bc.load(verse_cap=2000, char_cap=30000)
    names, rows, iso, script = d["names"], d["rows"], d["iso"], d["script"]
    fam_of = {r[0]: r[1] for r in rows}

    print(f"loaded {len(names)} langs, {len(d['common'])} aligned verses")
    print("transcribing to IPA (epitran, cached)...")
    ipa, kept, skipped = transcribe_cached(names, d["rawtext"], iso, script)

    print(f"\nIPA coverage: {len(kept)}/{len(names)} languages transcribed; "
          f"{len(skipped)} skipped")
    print("SKIPPED (graceful):")
    for lab, why in skipped:
        print(f"  - {lab:16s} {why}")

    # ---- work on the IPA-covered subset, same set for IPA and uroman ----------
    sub = kept
    sub_rows = [r for r in rows if r[0] in sub]
    sub_fam = {r[0]: r[1] for r in sub_rows}
    gold = lt.gold_newick_from_rows(sub_rows)

    ipa_clean = [lt.clean(ipa[n]) for n in sub]

    # uroman on the SAME subset (cache already on disk from the romanization arm)
    rom = bc.romanize_cached(sub, d["rawtext"], iso, tag=TAG)
    rom_clean = [lt.clean(rom[n]) for n in sub]
    raw_clean = [lt.clean(d["rawtext"][n]) for n in sub]

    def evaluate(texts, label):
        Djs = lt.js_matrix(texts, n=3)
        Dal = lt.alphabet_jaccard_matrix(texts)
        nwk = lt.linkage_to_newick(lt.upgma(Djs, sub), sub)
        diag = lt.nn_diagnostics(Djs, sub, sub_fam)
        tri = lt.rf_triple(nwk, gold, sub)
        g = lt.gqd(nwk, gold, sub)
        alpha_pur = lt.nn_purity(Dal, sub, sub_fam)
        alphabet = len(set("".join(texts)) - {" "})
        print(f"\n[{label}]  union alphabet/IPA-inventory = {alphabet} symbols")
        print(f"  family-NN purity   = {diag['purity']:.3f} "
              f"(tie=miss {diag['purity_tiemiss']:.3f}, {diag['n_ties']} ties, "
              f"chance floor {diag['chance']:.3f}, n={diag['n']})")
        print(f"  alphabet-baseline family-NN = {alpha_pur:.3f}  "
              f"(info-theory margin {diag['purity']-alpha_pur:+.3f})")
        print(f"  tree-vs-gold rf_triple: observed normRF={tri['observed']:.3f}, "
              f"floor={tri['floor']:.3f}, null_p50={tri['null_p50']:.3f}, "
              f"rescaled={tri['rescaled']:.3f}")
        print(f"  tree-vs-gold GQD     = {g['gqd']:.3f}  "
              f"(0 = correct/refinement; resolved_in_gold={g.get('resolved_in_gold')})")
        return dict(Djs=Djs, diag=diag, tri=tri, gqd=g, alpha_pur=alpha_pur,
                    alphabet=alphabet, nwk=nwk)

    print("\n" + "=" * 70)
    print(f"IPA vs uroman vs RAW on the SAME {len(sub)}-language IPA-covered subset")
    print("=" * 70)
    R = evaluate(raw_clean, "RAW (native scripts)")
    U = evaluate(rom_clean, "UROMAN (Latin, vowels dropped for abjad/abugida)")
    I = evaluate(ipa_clean, "IPA (epitran, vowels RESTORED)")

    # ---- direct cross-script test pairs: does IPA reconnect them? -------------
    print("\n" + "=" * 70)
    print("CROSS-SCRIPT TEST PAIRS — trigram-JS distance (lower = closer)")
    print("  RAW (native) -> UROMAN -> IPA")
    print("=" * 70)
    idx = {n: i for i, n in enumerate(sub)}
    # The three requested pairs + others that survive IPA coverage.
    requested = [("Hebrew", "Arabic"), ("Telugu", "Kannada"), ("Greek", "Latin")]
    extra = [("Telugu", "Malayalam"), ("Kannada", "Malayalam"),
             ("Russian", "Polish"), ("Russian", "Ukranian"),
             ("Farsi_Persian", "Arabic"), ("Hindi", "Marathi")]
    for tagp, pairs in [("REQUESTED", requested), ("ADDITIONAL (IPA-covered)", extra)]:
        print(f"\n  -- {tagp} --")
        for a, b in pairs:
            if a in idx and b in idx:
                ra = R["Djs"][idx[a], idx[b]]
                ua = U["Djs"][idx[a], idx[b]]
                ia = I["Djs"][idx[a], idx[b]]
                verdict = ("IPA closer" if ia < ua - 1e-6 else
                           "uroman closer" if ua < ia - 1e-6 else "tie")
                print(f"  {a:14s}-{b:12s}: raw {ra:.3f} -> uroman {ua:.3f} "
                      f"-> IPA {ia:.3f}   [{verdict}]")
            else:
                miss = [x for x in (a, b) if x not in idx]
                print(f"  {a:14s}-{b:12s}: SKIP — not IPA-covered ({', '.join(miss)})")

    # ---- figure: IPA tree coloured by family ---------------------------------
    fams = sorted(set(sub_fam.values()))
    cmap = plt.get_cmap("tab20")
    fam_color = {f: cmap(i % 20) for i, f in enumerate(fams)}
    from scipy.cluster.hierarchy import dendrogram
    Z = lt.upgma(I["Djs"], sub)
    fig, ax = plt.subplots(figsize=(11, 0.33 * len(sub) + 2))
    dendrogram(Z, labels=sub, orientation="right", ax=ax, color_threshold=0,
               above_threshold_color="#999")
    ax.set_title(f"IPA (epitran) parallel Bible — trigram Jensen-Shannon tree\n"
                 f"{len(sub)} languages, UPGMA, coloured by family — "
                 f"phonetic vowels restored")
    ax.set_xlabel("Jensen-Shannon distance (character trigrams over IPA)")
    for lbl in ax.get_ymajorticklabels():
        lbl.set_color(fam_color[sub_fam[lbl.get_text()]]); lbl.set_fontsize(8)
    ax.legend(handles=[Patch(facecolor=fam_color[f], label=f) for f in fams],
              loc="lower right", fontsize=6, ncol=2, framealpha=0.9)
    fig.tight_layout(); fig.savefig("figures/ipa_tree.png", dpi=130); plt.close(fig)
    print("\nsaved figures/ipa_tree.png")

    # ---- summary line --------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY (same IPA-covered subset)")
    print(f"  family-NN purity : RAW {R['diag']['purity']:.3f} | "
          f"uroman {U['diag']['purity']:.3f} | IPA {I['diag']['purity']:.3f} "
          f"(chance {I['diag']['chance']:.3f})")
    print(f"  rf_triple rescaled: RAW {R['tri']['rescaled']:.3f} | "
          f"uroman {U['tri']['rescaled']:.3f} | IPA {I['tri']['rescaled']:.3f}")
    print(f"  GQD vs gold      : RAW {R['gqd']['gqd']:.3f} | "
          f"uroman {U['gqd']['gqd']:.3f} | IPA {I['gqd']['gqd']:.3f}")
    print(f"done in {time.time()-t0:.0f}s")

    return dict(R=R, U=U, I=I, sub=sub, skipped=skipped)


if __name__ == "__main__":
    main()
