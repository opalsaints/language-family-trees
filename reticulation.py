"""Reticulation / treelikeness arm.

A single bifurcating tree forces ALL of the distance signal into nested splits.
Real language data carries non-tree signal: borrowing, areal diffusion, shared
sound changes across contact zones. This script quantifies how non-tree the
character-trigram JS signal is, using the Holland et al. (2002) delta score
(lt.delta_score): 0 = perfectly tree-like (every quartet additive), higher =
more reticulate / box-like.

We compute delta for:
  (1) the whole curated romanized set,
  (2) clean monophyletic sub-clades (Germanic, Romance, Slavic, Indo-Aryan),
  (3) known contact zones (Balkan Sprachbund), and a "spread" control
      (geographically dispersed Romance) for contrast.

Then we draw a NeighborNet (splitspy) of the whole set from the same JS matrix.
A NeighborNet shows non-tree signal as boxes/reticulations rather than clean
bifurcations; contact zones should appear as box-like regions.

Hypothesis: contact zones show HIGHER delta (more reticulate) than clean
single-genus sub-families.
"""
import os
import sys
import time
import traceback
import numpy as np

import langtree as lt
import biblecorpus as bc

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
N_GRAM = 3

t0 = time.time()
d = bc.load(verse_cap=2000, char_cap=30000)
names, rows, iso = d["names"], d["rows"], d["iso"]
fam_of = {r[0]: r[1] for r in rows}
genus_of = {r[0]: r[2] for r in rows}
subg_of = {r[0]: r[3] for r in rows}
idx = {n: i for i, n in enumerate(names)}

print(f"loaded {len(names)} langs, {len(d['common'])} aligned verses; "
      f"romanizing (cached)...")
rom = bc.romanize_cached(names, d["rawtext"], iso, tag="v2000_c30000")
rom_clean = {n: lt.clean(rom[n]) for n in names}
print(f"romanization ready ({time.time()-t0:.0f}s)\n")


def delta_for(labels, tag):
    """delta_score over the JS(trigram) matrix restricted to `labels`."""
    present = [l for l in labels if l in idx]
    missing = [l for l in labels if l not in idx]
    if len(present) < 4:
        return None, present, missing, None  # delta needs quartets
    texts = [rom_clean[l] for l in present]
    D = lt.js_matrix(texts, n=N_GRAM)
    delta = lt.delta_score(D)
    return delta, present, missing, D


# ---- sub-clade definitions from gold genus/subgenus ----
GERMANIC = [n for n in names if genus_of[n] == "Germanic"]
ROMANCE = [n for n in names if genus_of[n] == "Italic" and subg_of[n] == "Romance"]
SLAVIC = [n for n in names if genus_of[n] == "Slavic"]
INDO_ARYAN = [n for n in names if subg_of[n] == "Indo-Aryan"]
# Only 3 Indo-Aryan in corpus (Hindi, Marathi, Nepali); delta needs >=4 langs.
# Fall back to the next node up, Indo-Iranian genus (adds Farsi/Persian,
# Iranian branch), so we can still report a number for this clade.
INDO_IRANIAN = [n for n in names if genus_of[n] == "Indo-Iranian"]

# Balkan Sprachbund (classic contact zone): Greek (Hellenic), Bulgarian +
# Serbian (S. Slavic), Romanian (Romance), Albanian (Albanian). FIVE different
# IE branches that have converged areally -> should be highly reticulate.
BALKAN = ["Greek", "Bulgarian", "Romanian", "Albanian", "Serbian"]

# Control: a cross-genus IE mix with NO known shared contact zone, sampling one
# language from several distinct genera. If reticulation were just "more
# diversity = higher delta", this would also be high; if it's specifically
# contact, Balkan should exceed a comparable-diversity dispersed set.
DISPERSED_IE = ["Icelandic", "Portuguese", "Russian", "Hindi", "Lithuanian"]

CLADES = [
    ("WHOLE-SET", names, "all 57 (reference)"),
    ("Germanic", GERMANIC, "single genus, clean"),
    ("Romance", ROMANCE, "single genus, clean"),
    ("Slavic", SLAVIC, "single genus, clean"),
    ("Indo-Aryan", INDO_ARYAN, "single subgenus, clean (n=3, will skip)"),
    ("Indo-Iranian", INDO_IRANIAN, "genus fallback for Indo-Aryan, clean"),
    ("Balkan(contact)", BALKAN, "5 IE branches, areal Sprachbund"),
    ("Dispersed-IE(ctrl)", DISPERSED_IE, "5 IE branches, NO contact zone"),
]

print("=== DELTA SCORES (Holland 2002; 0 = tree-like, higher = reticulate) ===")
print(f"   {'clade':22s} {'n':>3s} {'delta':>7s}   note")
results = {}
for tag, labels, note in CLADES:
    delta, present, missing, D = delta_for(labels, tag)
    results[tag] = dict(delta=delta, present=present, missing=missing)
    if delta is None:
        print(f"   {tag:22s} {len(present):>3d}   SKIP  (need >=4 langs) {note}")
        if missing:
            print(f"        missing: {missing}")
    else:
        print(f"   {tag:22s} {len(present):>3d} {delta:7.4f}   {note}")
        if missing:
            print(f"        (missing from corpus, skipped: {missing})")

# ---- interpretation summary numbers ----
clean = [results[k]["delta"] for k in
         ("Germanic", "Romance", "Slavic", "Indo-Iranian")
         if results[k]["delta"] is not None]
clean_mean = float(np.mean(clean)) if clean else float("nan")
balkan = results["Balkan(contact)"]["delta"]
disp = results["Dispersed-IE(ctrl)"]["delta"]
whole = results["WHOLE-SET"]["delta"]

print("\n=== INTERPRETATION ===")
print(f"   mean delta over clean single-genus clades : {clean_mean:.4f}")
print(f"   Balkan contact zone delta                 : {balkan:.4f}")
print(f"   dispersed-IE control delta                : {disp:.4f}")
print(f"   whole-set delta                           : {whole:.4f}")
if balkan is not None:
    print(f"   Balkan vs clean-clade mean   : "
          f"{'HIGHER (more reticulate)' if balkan > clean_mean else 'lower'} "
          f"(+{balkan - clean_mean:+.4f})")
    print(f"   Balkan vs dispersed control  : "
          f"{'HIGHER' if balkan > disp else 'lower'} ({balkan - disp:+.4f})")

# ============================================================
# NeighborNet via splitspy on the whole-set JS distance matrix
# ============================================================
NN_PATH = os.path.join(FIGDIR, "neighbornet.png")
nn_status = "not attempted"
try:
    import splitspy.outline as sp_outline

    texts_all = [rom_clean[n] for n in names]
    D_all = lt.js_matrix(texts_all, n=N_GRAM)
    # splitspy wants python lists, not numpy
    matrix = [[float(x) for x in row] for row in D_all]
    labels = list(names)
    print(f"\n=== NeighborNet (splitspy) on whole-set JS({N_GRAM}-gram) ===")
    print(f"   {len(labels)} taxa -> {NN_PATH}")
    sp_outline.run(
        labels, matrix,
        outfile=NN_PATH,
        win_width=1400, win_height=1400,
        font_size=11,
    )
    if os.path.exists(NN_PATH) and os.path.getsize(NN_PATH) > 0:
        nn_status = f"OK ({os.path.getsize(NN_PATH)} bytes)"
        print(f"   wrote {NN_PATH} ({os.path.getsize(NN_PATH)} bytes)")
    else:
        nn_status = "FAILED: no/empty output file"
        print(f"   {nn_status}")
except Exception as e:
    nn_status = f"FAILED: {type(e).__name__}: {e}"
    print(f"\n=== NeighborNet FAILED ===\n   {nn_status}")
    traceback.print_exc()
    print("   (delta scores above are still valid; NeighborNet is the visual only.)")

print(f"\nNeighborNet status: {nn_status}")
print(f"done ({time.time()-t0:.0f}s)")
