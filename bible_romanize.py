"""Romanization arm (Stage: cross-script).

The raw character method is blind across writing systems (every unique-script
language sits at JS = 1.0 from everything). We romanize every language into one
Latin alphabet with uroman and re-run. Two questions:

  (1) Do cross-script families reconnect? (Hebrew-Arabic, Cyrillic/Latin Slavic,
      the Dravidian trio, ...)
  (2) The D1 "information-theory-must-work" test: once every language shares one
      alphabet, the alphabet-overlap baseline should COLLAPSE (inventories become
      near-identical), so any family recovery is the n-gram statistics doing the
      work — not letter inventory.
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import langtree as lt
import biblecorpus as bc

t0 = time.time()
d = bc.load(verse_cap=2000, char_cap=30000)
names, rows, iso, script = d["names"], d["rows"], d["iso"], d["script"]
fam_of = {r[0]: r[1] for r in rows}
gold = lt.gold_newick_from_rows(rows)

raw_clean = [lt.clean(d["rawtext"][n]) for n in names]
print(f"loaded {len(names)} langs, {len(d['common'])} verses; romanizing (cached)...")
rom = bc.romanize_cached(names, d["rawtext"], iso, tag="v2000_c30000")
rom_clean = [lt.clean(rom[n]) for n in names]
print(f"romanization ready ({time.time()-t0:.0f}s)")


def evaluate(texts, label):
    Djs = lt.js_matrix(texts, n=3)
    Dal = lt.alphabet_jaccard_matrix(texts)
    nwk = lt.linkage_to_newick(lt.upgma(Djs, names), names)
    _, _, rfn = lt.rf_corrected(nwk, gold)
    tri_pur = lt.nn_purity(Djs, names, fam_of)
    alpha_pur = lt.nn_purity(Dal, names, fam_of)
    alphabet = len(set("".join(texts)) - {" "})
    print(f"  {label:10s} | trigram-JS: normRF {rfn:.3f}, family-NN {tri_pur:.3f} "
          f"| alphabet-baseline family-NN {alpha_pur:.3f} | union alphabet {alphabet}")
    return dict(Djs=Djs, rfn=rfn, tri_pur=tri_pur, alpha_pur=alpha_pur, alphabet=alphabet)

print("\n=== RAW vs ROMANIZED ===")
R = evaluate(raw_clean, "RAW")
M = evaluate(rom_clean, "ROMANIZED")

print("\n=== cross-script family pairs: JS distance (raw -> romanized) ===")
idx = {n: i for i, n in enumerate(names)}
pairs = [("Hebrew", "Arabic"), ("Hebrew", "Amharic"), ("Russian", "Polish"),
         ("Telugu", "Kannada"), ("Greek", "Latin"), ("Russian", "Bulgarian")]
for a, b in pairs:
    if a in idx and b in idx:
        print(f"  {a:8s}-{b:9s}: {R['Djs'][idx[a],idx[b]]:.3f} -> {M['Djs'][idx[a],idx[b]]:.3f}")

print(f"\nD1 verdict (romanized, common-alphabet regime): "
      f"trigram-JS family-NN {M['tri_pur']:.3f} vs alphabet-baseline {M['alpha_pur']:.3f} "
      f"-> info-theory margin = {M['tri_pur']-M['alpha_pur']:+.3f}")

# ---- figures ----
fams = sorted(set(fam_of.values()))
cmap = plt.get_cmap("tab20")
fam_color = {f: cmap(i % 20) for i, f in enumerate(fams)}
from scipy.cluster.hierarchy import dendrogram
Z = lt.upgma(M["Djs"], names)
fig, ax = plt.subplots(figsize=(11, 0.33 * len(names) + 2))
dendrogram(Z, labels=names, orientation="right", ax=ax, color_threshold=0,
           above_threshold_color="#999")
ax.set_title(f"Romanized (uroman) parallel Bible — trigram Jensen–Shannon tree\n"
             f"{len(names)} languages, UPGMA, coloured by family — cross-script families reconnect")
ax.set_xlabel("Jensen–Shannon distance (character trigrams, romanized)")
for lbl in ax.get_ymajorticklabels():
    lbl.set_color(fam_color[fam_of[lbl.get_text()]]); lbl.set_fontsize(8)
ax.legend(handles=[Patch(facecolor=fam_color[f], label=f) for f in fams],
          loc="lower right", fontsize=6, ncol=2, framealpha=0.9)
fig.tight_layout(); fig.savefig("figures/bible_romanized_tree.png", dpi=130); plt.close(fig)
print("saved figures/bible_romanized_tree.png")

# D1 money figure: family-purity, trigram vs alphabet baseline, raw vs romanized
fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(2); w = 0.36
ax.bar(x - w/2, [R["tri_pur"], M["tri_pur"]], w, label="trigram-JS (information theory)", color="#1f77b4")
ax.bar(x + w/2, [R["alpha_pur"], M["alpha_pur"]], w, label="alphabet-overlap (dumb baseline)", color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(["RAW\n(many scripts)", "ROMANIZED\n(one alphabet)"])
ax.set_ylabel("nearest-neighbour family purity"); ax.set_ylim(0, 1)
ax.set_title("Romanizing equalizes the alphabet → the baseline collapses,\nso family recovery is the information theory doing the work")
for i, v in enumerate([R["tri_pur"], M["tri_pur"]]): ax.text(i - w/2, v + .02, f"{v:.2f}", ha="center", fontsize=8)
for i, v in enumerate([R["alpha_pur"], M["alpha_pur"]]): ax.text(i + w/2, v + .02, f"{v:.2f}", ha="center", fontsize=8)
ax.legend(fontsize=8, loc="upper center")
fig.tight_layout(); fig.savefig("figures/romanization_purity.png", dpi=130); plt.close(fig)
print("saved figures/romanization_purity.png")
print(f"done in {time.time()-t0:.0f}s")
