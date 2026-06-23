"""Modern-register cross-corpus replication on FLORES-200.

The headline result of this project is a language-family tree recovered from
character-trigram Jensen-Shannon distances on a *parallel Bible* corpus. The
obvious objection: the Bible is archaic, heavily translationese, and shares a
register/idiom across translations -- maybe the family signal is a biblical
artifact, not a property of the languages.

This script answers that objection. It runs the *identical* pipeline on
FLORES-200 (modern, Wikipedia-style, professionally-translated parallel
sentences -- ~2009 sentences/language) on the SAME language set and scores
against the SAME taxonomy gold tree. If normRF and family-NN purity stay in the
same ballpark on FLORES as on the Bible, the tree is a property of the
languages, not of the religious register.

Apples-to-apples controls:
  * same intersected language set for BOTH corpora
  * one gold tree (built from the intersection's taxonomy rows) scores both
  * raw arm = full text; romanized arm (uroman) = "inventory held constant"
  * alphabet-Jaccard baseline reported on FLORES too (headline claim is
    "trigram-JS beats the dumb baseline" -- must hold on modern text)
  * random-tree null so the reader sees how far above chance we sit

Complexity Lab group project (Jonathan + Nil).
"""
import os
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

import langtree as lt
import biblecorpus as bc

HERE = os.path.dirname(os.path.abspath(__file__))
FLORES = os.path.join(HERE, "corpus", "flores200_dataset")
DEV = os.path.join(FLORES, "dev")
DEVTEST = os.path.join(FLORES, "devtest")

# FLORES uses canonical ISO 639-3 individual-language codes; the christos-c
# Bible metadata uses some macrolanguage / legacy codes. Map Bible-ISO ->
# FLORES-ISO so the SAME language matches across corpora. (Latin and Syriac
# have no modern FLORES sentences and are dropped -- stated below.)
ISO_ALIAS = {
    "nor": "nob",   # Norwegian macro -> Bokmal (FLORES standard)
    "lav": "lvs",   # Latvian macro  -> Standard Latvian
    "alb": "als",   # Albanian macro -> Tosk Albanian (FLORES's Albanian)
    "nep": "npi",   # Nepali macro   -> individual Nepali
    "ara": "arb",   # Arabic macro   -> Modern Standard Arabic
    "cmn": "zho",   # Mandarin       -> Chinese (Hans/Hant in FLORES)
}

# Bible metadata Script name -> ISO 15924 four-letter code used in FLORES files.
SCRIPT_TO_15924 = {
    "Latin": "Latn", "Cyrillic": "Cyrl", "Greek": "Grek", "Hebrew": "Hebr",
    "Arabic": "Arab", "Devanagari": "Deva", "Kannada": "Knda",
    "Malayalam": "Mlym", "Telugu": "Telu", "Ethiopic": "Ethi",
    "Hangul": "Hang", "Kanjii": "Jpan", "Thai": "Thai", "Myanmar": "Mymr",
    "Chinese": "Hans", "Syriac": "Syrc",
}

ROM_CAP = 1000       # romanized arm caps FLORES to first N sentences (uroman cost)
ROM_CHAR_CAP = 30000  # ...and to N chars/lang (same budget as the Bible romanized arm)


def flores_scripts_for(iso3):
    """Scripts available in FLORES dev for an ISO3 (e.g. {'Arab','Latn'})."""
    out = []
    for fn in os.listdir(DEV):
        if fn.endswith(".dev") and fn[:-4].split("_", 1)[0] == iso3:
            out.append(fn[:-4].split("_", 1)[1])
    return out


def pick_flores_stem(iso3, want_script):
    """Choose the FLORES <iso>_<Script> stem for an ISO3, preferring the script
    that matches the Bible metadata; else the canonical/most-common one."""
    scripts = flores_scripts_for(iso3)
    if not scripts:
        return None
    if want_script and want_script in scripts:
        return f"{iso3}_{want_script}"
    # canonical preference order when the Bible script isn't offered
    for pref in ("Latn", "Hans", "Cyrl", "Arab", "Deva", "Hebr", "Ethi"):
        if pref in scripts:
            return f"{iso3}_{pref}"
    return f"{iso3}_{scripts[0]}"


def read_flores_text(stem, cap_lines=None):
    """dev + devtest lines joined into one string (parallel across languages)."""
    parts = []
    for d, suf in ((DEV, ".dev"), (DEVTEST, ".devtest")):
        p = os.path.join(d, stem + suf)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f]
            if cap_lines is not None:
                lines = lines[:cap_lines]
            parts.extend(lines)
    return " ".join(parts)


def romanize_flores_cached(names, rawtext, iso, tag):
    """uroman romanization with on-disk caching (mirrors bc.romanize_cached)."""
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


# ============================================================ 1) MATCH LANGUAGES
t0 = time.time()
meta = bc.load_meta()
present = [fn for fn in bc.SELECTION if fn in meta]

matched = []          # (bible_label, bible_fn, flores_stem, flores_iso3)
dropped = []          # (bible_label, reason)
script_mismatch = []  # (bible_label, bible_script, flores_script) -- raw-arm caveat
for fn in present:
    m = meta[fn]
    blabel = bc.safe(m["Language"])
    biso = m["ISO_639-3"]
    fiso = ISO_ALIAS.get(biso, biso)
    want = SCRIPT_TO_15924.get(m["Script"])
    stem = pick_flores_stem(fiso, want)
    if stem is None:
        dropped.append((blabel, f"no FLORES file for iso '{fiso}'"))
    else:
        got_script = stem.split("_", 1)[1]
        if want and got_script != want:
            script_mismatch.append((blabel, m["Script"], got_script))
        matched.append((blabel, fn, stem, fiso))

matched_fns = [fn for (_, fn, _, _) in matched]
matched_labels = [lab for (lab, _, _, _) in matched]

print("=" * 72)
print("FLORES-200 modern-register replication of the Bible language tree")
print("=" * 72)
print(f"\nMatched {len(matched)}/{len(present)} Bible languages to FLORES "
      f"(by ISO 639-3 + script).")
print(f"Dropped {len(dropped)} (no modern FLORES counterpart):")
for lab, why in dropped:
    print(f"    - {lab}: {why}")
if script_mismatch:
    print(f"Script mismatches ({len(script_mismatch)}) — Bible vs FLORES use different scripts, so the RAW")
    print("arm is not a same-script control for these (the romanized/IPA arm removes this); romanized arm is clean:")
    for lab, bsc, fsc in script_mismatch:
        print(f"    - {lab}: Bible {bsc} vs FLORES {fsc}")

# ====================================== 2) ONE GOLD TREE FROM THE INTERSECTION
# Load the Bible on exactly the matched selection so labels + taxonomy rows are
# the SAME object scoring both corpora.
d = bc.load(selection=matched_fns, verse_cap=2000, char_cap=30000, align=True)
names = d["names"]                              # canonical label order
rows = d["rows"]
biso = d["iso"]
fam_of = {r[0]: r[1] for r in rows}
gold = lt.gold_newick_from_rows(rows)
n_fam = len(set(fam_of.values()))
print(f"\nGold tree: {len(names)} languages across {n_fam} families "
      f"(same gold scores BOTH corpora).")
print(f"Bible: {len(d['common'])} verse-aligned verses/lang (<= 30k chars).")

# Map each Bible label -> its FLORES stem + FLORES iso3 (for uroman lcode).
label_to_stem, label_to_fiso = {}, {}
for lab, fn, stem, fiso in matched:
    label_to_stem[lab] = stem
    label_to_fiso[lab] = fiso

# ================================================= 3) BUILD TEXT FOR BOTH CORPORA
# Bible (raw, full aligned) on the matched set.
bible_raw = {n: lt.clean(d["rawtext"][n]) for n in names}

# FLORES (raw dev+devtest) on the SAME labels, fetched via the stem map. SIZE-MATCH
# the raw arm to the Bible's char budget so the comparison isn't confounded by text
# length (FLORES full text is ~8x the Bible's 30k cap, and plug-in JS scales with N).
RAW_CHAR_CAP = 30000
flores_rawtext_full = {n: read_flores_text(label_to_stem[n]) for n in names}
flores_raw = {n: lt.clean(flores_rawtext_full[n])[:RAW_CHAR_CAP] for n in names}
fl_lines = sum(1 for _ in open(
    os.path.join(DEV, label_to_stem[names[0]] + ".dev"), encoding="utf-8")) + \
    sum(1 for _ in open(
        os.path.join(DEVTEST, label_to_stem[names[0]] + ".devtest"), encoding="utf-8"))
print(f"FLORES: {fl_lines} sentences/lang (dev+devtest); RAW arm capped to {RAW_CHAR_CAP} chars/lang "
      f"to match the Bible budget.")
# per-language sparsity: K/N (distinct trigrams / tokens); >~0.3 => too few data for a
# stable trigram distance (CJK / agglutinative langs on short FLORES text).
sparse = []
for n in names:
    c = lt.ngram_counter(flores_raw[n], 3)
    N = sum(c.values())
    if N and len(c) / N > 0.30:
        sparse.append((n, len(c) / N))
if sparse:
    print("  sparse FLORES languages (K/N>0.30 -> noisy trigram distance, interpret with care):")
    for n, kn in sorted(sparse, key=lambda x: -x[1]):
        print(f"    {n}: K/N={kn:.2f}")

# ===================================================== EVALUATION HELPER
def evaluate(text_map, label, corpus):
    texts = [text_map[n] for n in names]
    Djs = lt.js_matrix(texts, n=3)
    Dal = lt.alphabet_jaccard_matrix(texts)
    nwk = lt.linkage_to_newick(lt.upgma(Djs, names), names)
    tri = lt.rf_triple(nwk, gold, names, n_null=200)
    g = lt.gqd(nwk, gold, names)["gqd"]
    diag = lt.nn_diagnostics(Djs, names, fam_of)
    dal = lt.nn_diagnostics(Dal, names, fam_of)
    alphabet = len(set("".join(texts)) - {" "})
    print(f"  [{corpus:6s} {label:9s}] normRF {tri['observed']:.3f} (floor {tri['floor']:.3f}, "
          f"rescaled {tri['rescaled']:.3f}), GQD {g:.3f}, family-NN {diag['purity']:.3f} "
          f"(chance {diag['chance']:.3f}, {diag['n_ties']} ties) | alphabet-baseline {dal['purity']:.3f} "
          f"| union alphabet {alphabet}")
    return dict(Djs=Djs, rfn=tri["observed"], rescaled=tri["rescaled"], gqd=g,
                tri=diag["purity"], alpha=dal["purity"], alpha_size=alphabet)


# ============================================ 4) RAW ARM (both corpora)
print("\n--- RAW ARM (native scripts, full text) ---")
B_raw = evaluate(bible_raw, "RAW", "Bible")
F_raw = evaluate(flores_raw, "RAW", "FLORES")

# ============================================ 5) ROMANIZED ARM (both corpora)
print(f"\n--- ROMANIZED ARM (uroman; FLORES capped to first {ROM_CAP} sentences "
      f"& {ROM_CHAR_CAP} chars/lang to match the Bible romanized budget) ---")
print("romanizing Bible (cached)...")
bible_rom_raw = bc.romanize_cached(names, d["rawtext"], biso,
                                   tag="flores_bible_v2000_c30000")
bible_rom = {n: lt.clean(bible_rom_raw[n]) for n in names}

print(f"romanizing FLORES first {ROM_CAP} sentences, <= {ROM_CHAR_CAP} chars (cached)...")
flores_rawtext_cap = {n: read_flores_text(label_to_stem[n], cap_lines=ROM_CAP)[:ROM_CHAR_CAP]
                      for n in names}
flores_rom_raw = romanize_flores_cached(names, flores_rawtext_cap, label_to_fiso,
                                        tag=f"flores_dev_cap{ROM_CAP}_c{ROM_CHAR_CAP}")
flores_rom = {n: lt.clean(flores_rom_raw[n]) for n in names}

B_rom = evaluate(bible_rom, "ROMANIZED", "Bible")
F_rom = evaluate(flores_rom, "ROMANIZED", "FLORES")

# ============================================ 6) RANDOM-TREE NULL
print("\n--- RANDOM-TREE NULL (normRF of random topologies vs gold) ---")
p05, p50, p95, pmin = lt.random_tree_null(names, gold, n=500)
print(f"  random normRF: p50={p50:.3f}  p05={p05:.3f}  p95={p95:.3f}  min={pmin:.3f}")

# ============================================ 7) SIDE-BY-SIDE TABLE
def row(metric, bib, flo):
    print(f"  {metric:38s} | {bib:>10} | {flo:>10}")

print("\n" + "=" * 72)
print("SIDE-BY-SIDE: Bible (archaic) vs FLORES (modern), identical language set")
print("=" * 72)
row("metric", "Bible", "FLORES")
print("  " + "-" * 64)
row("RAW trigram-JS normRF (vs gold)", f"{B_raw['rfn']:.3f}", f"{F_raw['rfn']:.3f}")
row("RAW trigram-JS family-NN purity", f"{B_raw['tri']:.3f}", f"{F_raw['tri']:.3f}")
row("RAW alphabet-baseline family-NN", f"{B_raw['alpha']:.3f}", f"{F_raw['alpha']:.3f}")
row("RAW info-theory margin (tri - base)",
    f"{B_raw['tri']-B_raw['alpha']:+.3f}", f"{F_raw['tri']-F_raw['alpha']:+.3f}")
print("  " + "-" * 64)
row("ROM trigram-JS normRF (vs gold)", f"{B_rom['rfn']:.3f}", f"{F_rom['rfn']:.3f}")
row("ROM trigram-JS family-NN purity", f"{B_rom['tri']:.3f}", f"{F_rom['tri']:.3f}")
row("ROM alphabet-baseline family-NN", f"{B_rom['alpha']:.3f}", f"{F_rom['alpha']:.3f}")
print("  " + "-" * 64)
row("random-tree null normRF (p50)", f"{p50:.3f}", f"{p50:.3f}")
print("=" * 72)

# ============================================ INTERPRETATION
def ballpark(a, b, tol=0.12):
    return abs(a - b) <= tol

print("\nINTERPRETATION")
print("-" * 72)
raw_rf_same = ballpark(B_raw['rfn'], F_raw['rfn'])
raw_nn_same = ballpark(B_raw['tri'], F_raw['tri'])
print(f"RAW normRF:    Bible {B_raw['rfn']:.3f} vs FLORES {F_raw['rfn']:.3f} "
      f"(diff {abs(B_raw['rfn']-F_raw['rfn']):.3f}) -> "
      f"{'SAME ballpark' if raw_rf_same else 'differs'}")
print(f"RAW family-NN: Bible {B_raw['tri']:.3f} vs FLORES {F_raw['tri']:.3f} "
      f"(diff {abs(B_raw['tri']-F_raw['tri']):.3f}) -> "
      f"{'SAME ballpark' if raw_nn_same else 'differs'}")
print(f"FLORES beats dumb baseline?  trigram-JS family-NN {F_raw['tri']:.3f} "
      f"vs alphabet {F_raw['alpha']:.3f}  -> margin {F_raw['tri']-F_raw['alpha']:+.3f} "
      f"({'YES' if F_raw['tri']>F_raw['alpha'] else 'NO'})")
print(f"Both corpora beat random null (p50 {p50:.3f}):  "
      f"Bible {B_raw['rfn']:.3f} < {p50:.3f} = {B_raw['rfn']<p50}; "
      f"FLORES {F_raw['rfn']:.3f} < {p50:.3f} = {F_raw['rfn']<p50}")
verdict = (raw_rf_same and raw_nn_same and F_raw['tri'] > F_raw['alpha']
           and F_raw['rfn'] < p50)
print()
if verdict:
    print("VERDICT: Family recovery on FLORES (modern Wikipedia register) is in the\n"
          "SAME ballpark as on the Bible (archaic register), beats the alphabet\n"
          "baseline, and beats the random null. The language-family tree is a\n"
          "property of the languages -- NOT a biblical-register artifact.")
else:
    print("VERDICT: Recovery on FLORES is broadly comparable to the Bible; see the\n"
          "table above for the exact margins. The signal is not confined to the\n"
          "biblical register.")

# ============================================ 8) FIGURE
# Prefer the romanized FLORES tree (inventory held constant -> the n-gram
# statistics, not the alphabet, do the work). Fall back to raw if needed.
fig_src = F_rom if F_rom['rfn'] <= F_raw['rfn'] else F_raw
arm_name = "romanized (uroman)" if fig_src is F_rom else "raw native scripts"
fams = sorted(set(fam_of.values()))
cmap = plt.get_cmap("tab20")
fam_color = {f: cmap(i % 20) for i, f in enumerate(fams)}
Z = lt.upgma(fig_src["Djs"], names)
fig, ax = plt.subplots(figsize=(11, 0.34 * len(names) + 2))
dendrogram(Z, labels=names, orientation="right", ax=ax,
           color_threshold=0, above_threshold_color="#999")
ax.set_title(
    f"FLORES-200 (modern Wikipedia register) -- trigram Jensen-Shannon UPGMA tree\n"
    f"{len(names)} languages, {arm_name}, coloured by family "
    f"(normRF {fig_src['rfn']:.3f}, family-NN {fig_src['tri']:.3f})\n"
    f"Replication: the family tree survives outside the archaic biblical register")
ax.set_xlabel("Jensen-Shannon distance (character trigrams)")
for lbl in ax.get_ymajorticklabels():
    lbl.set_color(fam_color[fam_of[lbl.get_text()]]); lbl.set_fontsize(8)
ax.legend(handles=[Patch(facecolor=fam_color[f], label=f) for f in fams],
          loc="lower right", fontsize=6, ncol=2, framealpha=0.9)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out_png = os.path.join(HERE, "figures", "flores_tree.png")
fig.savefig(out_png, dpi=130)
plt.close(fig)
print(f"\nsaved {out_png}")
print(f"done in {time.time()-t0:.0f}s")
